import pickle
from collections import Counter
import math
import numpy as np
import random

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = None


UNK_token = '<UNK>'
PAD_token = '<PAD>'
SOS_token = '<SOS>'
EOS_token = '<EOS>'


split_level = 'word' # char, word

def tokenizer(text):
	if split_level == 'char':
		return [tok for tok in list(text)]
	elif  split_level == 'word':
		return [tok for tok in text.split(' ')]

def detokenizer(tokens):
	if split_level == 'char':
		return ''.join(tokens)
	elif  split_level == 'word':
		return ' '.join(tokens)


class Vocab(object):

	def __init__(self, reserved=[]):
		self.word2index = {}
		self.word2count = Counter()
		self.reserved = [PAD_token, UNK_token] + reserved
		self.index2word = list()
		self.embeddings = None


	def add_words(self, words):
		self.word2count.update(words)


	def build(self, vocab_size=None, min_freq=1):

		# sort by frequency, then alphabetically
		word_frequencies = sorted(self.word2count.items(), key=lambda tup: tup[0])
		word_frequencies.sort(key=lambda tup: tup[1], reverse=True)
		if vocab_size:
			word_frequencies = word_frequencies[:vocab_size]

		self.word2count = Counter()
		self.index2word = self.reserved[:]
		self.word2index.update({tok: i for i, tok in enumerate(self.index2word)})

		for word, freq in word_frequencies:
			if freq < min_freq: break
			self.word2index[word] = len(self.index2word)
			self.word2count[word] = freq
			self.index2word.append(word)


	def load(self, file_name):
		with open(file_name, 'r') as fin:
			for line in fin:
				line=line.strip()
				word, freq = line.split('\t')
				self.word2count[word] = freq


	def save(self, file_name):
		with open(file_name, 'w') as fout:

			for word, freq in self.word2count.most_common():
				fout.write('{}\t{}\n'.format(word, freq))


class RawField():
	
	def preprocess(self, text):
		return text

	def process(self, batch, device=None):
		return batch


	def save(self, file_name):
		with open(file_name, 'wb') as fout:
			pickle.dump(self, fout)

	def load(self, file_name):
		with open(file_name, 'rb') as fin:
			self.__dict__.update(pickle.load(fin).__dict__)



class TokenizedField(RawField):
	def preprocess(self, text):
		return tokenizer(text)


class Field(RawField):

	def __init__(self, add_sos=False, add_eos=False, tokenizer=tokenizer):
		self.add_sos = add_sos
		self.add_eos = add_eos
		self.tokenizer = tokenizer


	def preprocess(self, text):
		return self.tokenizer(text)


	def process(self, batch, device=None):
		padded, lengths = self.pad(batch)
		idx = self.numericalize(padded)
		tensor, lengths = self.to_tensor(idx, lengths, device)
		return tensor, lengths
		# return idx, lengths


	def numericalize(self, padded):
		idx = [[self.vocab.word2index[tok] for tok in ex] for ex in padded]
		return idx


	def to_tensor(self, idx, lengths, device):
		lengths = torch.tensor(lengths, dtype=torch.long, device=device)
		tensor = torch.tensor(idx, dtype=torch.long, device=device)
		tensor = tensor.contiguous()
		return tensor, lengths


	def pad(self, batch):
		max_len = max(len(x) for x in batch)
		padded, lengths = [], []
		
		for x in batch:
			padded.append(
				([] if self.add_sos is False else [SOS_token]) +
				list(x[:max_len]) +
				([] if self.add_eos is False else [EOS_token]) +
				[PAD_token] * max(0, max_len - len(x)))

			lengths.append(len(padded[-1]) - max(0, max_len - len(x)))

		return (padded, lengths)


	def build_vocab(self, datasets, vocab_size=None, min_freq=1, load_path=None, save_path=None):

		reserved = []
		if self.add_sos: reserved.append(SOS_token)
		if self.add_eos: reserved.append(EOS_token)

		self.vocab = Vocab(reserved=reserved)

		if load_path:
			self.vocab.load(load_path)
		else:
			for dataset in datasets:
				for example in dataset:
					self.vocab.add_words(example)

		self.vocab.build(vocab_size, min_freq)

		if save_path:
			self.vocab.save(save_path)





class Example():

	def __init__(self, data, fields):

		for (name, field), val in zip(fields, data):
			setattr(self, name, field.preprocess(val))


class Dataset():

	def __init__(self, examples=None, fields=None):
		if examples: self.examples = examples
		if fields: self.fields = dict(fields)


	def save(self, dataset_file):
		with open(dataset_file, 'wb') as fout:
			pickle.dump(self.examples, fout)


	def load(self, dataset_file, fields):
		with open(dataset_file, 'rb') as fin:
			self.examples = pickle.load(fin)
		self.fields = dict(fields)


	def sort(self, sort_key):
		sorted_examples = sorted(self.examples, key=sort_key)
		return sorted_examples


	def __getitem__(self, i):
		return self.examples[i]

	def __len__(self):
		try:
			return len(self.examples)
		except TypeError:
			return 2**32

	def __iter__(self):
		for x in self.examples:
			yield x

	def __getattr__(self, attr):
		for x in self.examples:
			yield getattr(x, attr)




class Batch():

	def __init__(self, data=None, dataset=None, device=None):

		self.batch_size = len(data)
		self.fields = dataset.fields

		for (name, field) in self.fields.items():
			if field is not None:
				batch = [getattr(x, name) for x in data]
				setattr(self, name, field.process(batch, device))

	def __len__(self):
		return self.batch_size




class BucketedDataIterator():

	def __init__(self, dataset, batch_size, sort_key, device=None, 
				repeat=False, shuffle=None, sort=None, num_buckets=100):
	   
		assert num_buckets < len(dataset), 'num_buckets must be smaller than dataset length.'

		self.dataset = dataset
		self.batch_size = batch_size+1
		self.sort_key = sort_key
		self.device = device
		self.repeat = repeat
		self.shuffle = shuffle
		self.sort = sort
		self.num_buckets = num_buckets
		
		self.iterations = 0
		self.epoch_iterations = 0


	def create_buckets(self):

		sorted_dataset = self.dataset.sort(self.sort_key)
		self.size = math.floor(len(sorted_dataset) / self.num_buckets)
		self.buckets = []
		
		for bucket in range(self.num_buckets):
			self.buckets.append(sorted_dataset[bucket*self.size: (bucket+1)*self.size - 1])
		
		# cursor[i] will be the cursor for the ith bucket
		self.cursor = np.array([0] * self.num_buckets)
		self.random_shuffler()
	

	def init_epoch(self):

		self.create_buckets()

		self.epoch_iterations = 0

		if not self.repeat:
			self.iterations = 0

	def __len__(self):
		return math.ceil(len(self.dataset) / self.batch_size)


	@property
	def epoch(self):
		return math.floor(self.iterations / len(self))


	def random_shuffler(self):
		#sorts buckets by sequence length, but keeps it random within the same length
		for i in range(self.num_buckets):
			random.shuffle(self.buckets[i])
			self.cursor[i] = 0


	def __iter__(self):

		while True:

			self.init_epoch()

			for idx in range(len(self.dataset)//self.batch_size):

				self.iterations += 1
				self.epoch_iterations += 1

				if np.any(self.cursor+self.batch_size+1 > self.size):
					self.random_shuffler()

				i = np.random.randint(0,self.num_buckets)

				batch = self.buckets[i][self.cursor[i]:self.cursor[i]+self.batch_size-1]
				self.cursor[i] += self.batch_size

				yield Batch(batch, self.dataset, self.device)
			
			if not self.repeat:
				return

###############################################################################################################



class TextDataset(Dataset):


	def __init__(self, fields, src_path, trg_path, src_max_length=None, trg_max_length=None):

		fields = [('src', fields[0]), ('trg', fields[1]), ('src_raw', fields[2])]

		examples = []

		with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
			for line_i, (src_line, trg_line) in enumerate(zip(src_file, trg_file)):
				src, trg = tokenizer(src_line.strip()), tokenizer(trg_line.strip())
				print('Reading line #: ', line_i, end='\r')

				if src_max_length is not None:
					src = src[:src_max_length]
					src = detokenizer(src)

				if trg_max_length is not None:
					trg = trg[:trg_max_length]
					trg = detokenizer(trg)

				if src != '' and trg != '':
					examples.append(Example([src, trg, src], fields))

		super(TextDataset, self).__init__(examples, fields)

