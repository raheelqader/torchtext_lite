import pickle
from collections import Counter
import math
import numpy as np
import random
from collections import defaultdict 
import warnings
import json
import os
from itertools import chain

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


UNK_token = '<UNK>'
PAD_token = '<PAD>'
SOS_token = '<SOS>'
EOS_token = '<EOS>'


split_level = 'char' # char, word

def tokenizer(text):
	if split_level == 'char':
		return [tok for tok in list(text)]
	elif  split_level == 'word':
		return [tok for tok in text.split()]

def detokenizer(tokens):
	if split_level == 'char':
		return ''.join(tokens)
	elif  split_level == 'word':
		return ' '.join(tokens)


class Vocab(object):

	def __init__(self, reserved=[]):
		self.reserved = [PAD_token, UNK_token] + reserved
		self.word2index = defaultdict(self._unk_idx)
		self.word2count = Counter()
		self.index2word = list()
		self.embeddings = None

	#returns location of UNK_token
	def _unk_idx(self):
		return self.reserved.index(UNK_token)


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
				self.word2count[word] = int(freq)

	def save(self, file_name):
		with open(file_name, 'w') as fout:

			for word, freq in self.word2count.most_common():
				fout.write('{}\t{}\n'.format(word, freq))



class RawField():
	
	def preprocess(self, data):
		return data

	def process(self, batch, device):
		return batch

	def save(self, file_name):
		with open(file_name, 'wb') as fout:
			pickle.dump(self, fout)

	def load(self, file_name):
		with open(file_name, 'rb') as fin:
			self.__dict__.update(pickle.load(fin).__dict__)


class AudioField(RawField):

	def process(self, batch, device):
		padded, lengths = self.zero_pad(batch)
		tensor, lengths = self.to_tensor(padded, lengths, device)
		return tensor, lengths

	def to_tensor(self, batch, lengths, device):
		lengths = torch.tensor(lengths, dtype=torch.long, device=device)
		tensor = torch.tensor(batch, dtype=torch.float, device=device)
		tensor = tensor.contiguous()
		return tensor, lengths

	def zero_pad(self, batch):
		max_len = max(len(x) for x in batch)
		padded, lengths = [], []
		pad_token = np.zeros_like(batch[-1][-1])

		for x in batch:
			padding_length = max(0, max_len - len(x))
			padded.append(np.concatenate((x, np.tile(pad_token, (padding_length, 1))), axis=0))
			lengths.append(len(padded[-1]) - max(0, max_len - len(x)))

		return (padded, lengths)

	def save(self, file_name):
		with open(file_name, 'wb') as fout:
			pickle.dump(self, fout)

	def load(self, file_name):
		with open(file_name, 'rb') as fin:
			self.__dict__.update(pickle.load(fin).__dict__)


class TextField(RawField):

	def __init__(self, add_sos=False, add_eos=False, tokenizer=tokenizer):
		self.add_sos = add_sos
		self.add_eos = add_eos
		self.tokenizer = tokenizer

	def preprocess(self, text):
		return self.tokenizer(text)

	def process(self, batch, device):
		padded, lengths = self.pad(batch)
		idx = self.numericalize(padded)
		tensor, lengths = self.to_tensor(idx, lengths, device)
		return tensor, lengths

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
			if not isinstance(datasets, list): datasets = [datasets]
			
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

	def save(self, file_name):
		with open(file_name, 'wb') as fout:
			dataset = {'examples':self.examples, 'fields':self.fields}
			pickle.dump(dataset, fout)

	def load(self, file_name):
		with open(file_name, 'rb') as fin:
			dataset = pickle.load(fin)
		self.examples = dataset['examples']
		self.fields = dataset['fields']

	def sort(self, sort_key, reverse=False):
		sorted_examples = sorted(self.examples, key=sort_key, reverse=reverse)
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

	def __init__(self, data, dataset, device):
		self.data = data
		self.batch_size = len(data)
		self.fields = dataset.fields
		self.device = device
		self.prepare_batch(data)

	def prepare_batch(self, data):
		for (name, field) in self.fields.items():
			if field is not None:
				batch = [getattr(x, name) for x in data]
				setattr(self, name, field.process(batch, self.device))

	def sort(self, sort_key, reverse=False):
		sorted_data = sorted(self.data, key=sort_key, reverse=reverse)
		self.prepare_batch(sorted_data)

	def __len__(self):
		return self.batch_size



class BucketIterator():

	def __init__(self, dataset, batch_size, sort_key, device=None, repeat=False, 
				shuffle=False, sort=False, sort_within_batch=True, num_buckets=None):
	   
		self.dataset = dataset
		self.batch_size = batch_size
		self.sort_key = sort_key
		self.device = device
		self.repeat = repeat
		self.shuffle = shuffle
		self.sort = sort
		self.sort_within_batch = sort_within_batch
		self.num_buckets = num_buckets or max(len(self.dataset)//(self.batch_size*100), 1)

		self.iterations = 0
		self.epoch_iterations = 0

		self.create_buckets()


	def create_buckets(self):
		
		if (len(self.dataset)//self.num_buckets) < self.batch_size:
			self.num_buckets = 1
			warnings.warn("num_buckets is set to 1 as the provided num_buckets is larger than the batch_size.")

		if self.sort:
			self.dataset = self.dataset.sort(self.sort_key)

		self.size = math.floor(len(self.dataset) / self.num_buckets)
		self.buckets = []

		for bucket in range(self.num_buckets):
			self.buckets.append(self.dataset[bucket*self.size: (bucket+1)*self.size])
		
		# cursor[i] will be the cursor for the ith bucket
		self.cursor = np.array([0] * self.num_buckets)
		

	@property
	def batches(self):
		"""
		if dataset is sorted, prepare batches from random buckets
		otherwise iterate over the buckets sequentially to prepare batches
		"""
		if self.sort:

			for idx in range(len(self.dataset)//self.batch_size):

				if np.any(self.cursor+self.batch_size+1 > self.size):
					self.random_shuffler()

				i = np.random.randint(0,self.num_buckets)

				batch = self.buckets[i][self.cursor[i]:self.cursor[i]+self.batch_size]
				self.cursor[i] += self.batch_size

				yield Batch(batch, self.dataset, self.device)
		else:
			batch = []
			for item in chain(*self.buckets):
				batch.append(item)
				if len(batch) == self.batch_size:
					yield Batch(batch, self.dataset, self.device)
					batch = []
			if batch:
				yield Batch(batch, self.dataset, self.device)


	def random_shuffler(self):
		#sorts buckets by sequence length, but keeps it random within the same length
		for i in range(self.num_buckets):
			if self.shuffle:
				random.shuffle(self.buckets[i])
			self.cursor[i] = 0


	def init_epoch(self):

		self.random_shuffler()
		self.epoch_iterations = 0

		if not self.repeat:
			self.iterations = 0


	def __len__(self):
		return math.ceil(len(self.dataset) / self.batch_size)


	@property
	def epoch(self):
		return math.floor(self.iterations / len(self))


	def __iter__(self):

		while True:

			self.init_epoch()

			for batch in self.batches:

				self.iterations += 1
				self.epoch_iterations += 1

				if self.sort_within_batch:
					batch.sort(sort_key=self.sort_key, reverse=True)

				yield batch
			
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




class ASRDataset(Dataset):


	def __init__(self, fields, base_path, data_path):

		fields = [('speech', fields[0]), ('text', fields[1]), ('src_text', fields[2]), ('trg_text', fields[3])]

		examples = []

		with open(data_path, encoding='utf-8') as data_file:
			for line_i, data_line in enumerate(data_file):
				json_data = json.loads(data_line)
				
				text = json_data["text"]

				audio = os.path.join(base_path, json_data["audio"])
				mfcc = np.load(audio.replace('.wav', '.npy'))

				print('Reading line #: ', line_i, end='\r')

				if text != '' and mfcc is not None:
					examples.append(Example([mfcc, text, text, text], fields))

		super(ASRDataset, self).__init__(examples, fields)
