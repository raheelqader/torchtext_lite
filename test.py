from torchtext_lite import *
# import torchtext

src_path = './source.txt'
trg_path = './target.txt' 
src_max_length=100
trg_max_length=100

from time import time
last_time = time()
def cal_time(msg):
	global last_time
	current_time = time()
	seconds_elapsed = current_time - last_time
	print(msg, ' seconds_elapsed: ', seconds_elapsed)
	last_time = time()


def test1():

	dataset = TextDataset()
	dataset.build_dataset(src_path, trg_path, src_max_length, trg_max_length)
	dataset.build_vocab(share_vocab=True, src_vocab_path='vocab_file', save=True)
	# dataset.build_vocab(share_vocab=False, src_vocab_path='src_vocab_file', trg_vocab_path='trg_vocab_file', save=True)
	dataset.save('dataset')

def test2():
	dataset = TextDataset()
	dataset.load('dataset', )
	# dataset.build_vocab(vocab_size=50, share_vocab=True, src_vocab_path='vocab_file', save=True)
	dataset.build_vocab(vocab_size=50, share_vocab=False, src_vocab_path='src_vocab_file', trg_vocab_path='trg_vocab_file', save=True)
	print(sorted(dataset.src_vocab.word2index.items(), key= lambda x:x[1]))
	print('*'*20)
	print(sorted(dataset.trg_vocab.word2index.items(), key= lambda x:x[1]))


def test3():
	src_field = Field()
	trg_field = Field()
	dataset = TextDataset(src_field, trg_field, src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src,dataset.trg), save_path='vocab_file')
	dataset.save('dataset')

def test4():
	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()
	dataset = TextDataset(src_field, trg_field, src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src,dataset.trg), vocab_size=50000, save_path='vocab_file')
	dataset.save('dataset')

	# src_field.process(dataset[:2])
	
	src_examples = []
	counter = 0
	for x in dataset.src:
		src_examples.append(x)
		counter+=1
		if counter>=4: break
	

	src_field.process(src_examples)


def test5():
	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()

	dataset = TextDataset((src_field, trg_field), src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src, dataset.trg), vocab_size=50000, save_path='vocab_file')
	dataset.save('dataset')

	# src_field.process(dataset[:2])
	
	src_examples = []
	counter = 0
	for x in dataset.src:
		src_examples.append(x)
		counter+=1
		if counter>=100: break
	

	print(src_field.process(src_examples))



def test6():

	src_field = Field()
	trg_field = Field()
	fields = [('src', src_field), ('trg', trg_field)]

	dataset = Dataset()
	dataset.load(dataset_file='./dataset', fields=fields)

	src_field.build_vocab((dataset.src, dataset.trg), vocab_size=50000)
	src_field.save('src_field')

	src_examples = []
	counter = 0
	for x in dataset.src:
		src_examples.append(x)
		counter+=1
		if counter>=100: break
	

	# print(src_field.process(src_examples))



def test7():

	src_field = Field()
	trg_field = Field()
	fields = [('src', src_field), ('trg', trg_field)]

	dataset = Dataset()
	dataset.load(dataset_file='./dataset', fields=fields)

	src_field.load('src_field')

	src_examples = []
	counter = 0
	for x in dataset.src:
		src_examples.append(x)
		counter+=1
		if counter>=4: break
	

	print(src_field.process(src_examples))



def test8():
	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()

	dataset = TextDataset((src_field, trg_field), src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src, dataset.trg), vocab_size=50000, save_path='vocab_file')


	src_examples = []
	counter = 0
	for x in dataset.src:
		src_examples.append(x)
		counter+=1
		if counter>=12: break
	
	print(src_field.process(src_examples))



def test9():
	def tokenizer(text):
		return text.split('al')
		
	src_field = Field(add_sos=True, add_eos=True, tokenizer=tokenizer)
	trg_field = Field()

	dataset = TextDataset((src_field, trg_field), src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src, dataset.trg), vocab_size=50000, save_path='vocab_file')
	src_field.save('src_field')

	src_examples = []
	counter = 0
	for x in dataset.src:
		src_examples.append(x)
		counter+=1
		if counter>=12: break
	
	print(src_field.process(src_examples))



def test10():

	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()
	src_raw_field = RawField()

	dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src,), vocab_size=50000, save_path='src-vocab_file')
	trg_field.build_vocab((dataset.trg,), vocab_size=50000, save_path='trg-vocab_file')

	src_examples = []
	counter = 0
	for x in dataset.trg:
		src_examples.append(x)
		counter+=1
		if counter>=12: break
	
	print(trg_field.process(src_examples))


def test11():

	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()
	src_raw_field = RawField()

	dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src,), vocab_size=50000, save_path='src-vocab_file')
	trg_field.build_vocab((dataset.trg,), vocab_size=50000, save_path='trg-vocab_file')

	src_raw_examples = []
	counter = 0
	for x in dataset.src_raw:
		src_raw_examples.append(x)
		counter+=1
		if counter>=2: break
		print('*'*5)
	
	print(src_raw_field.process(src_raw_examples))




def test12():
	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()
	src_raw_field = RawField()

	dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src, dataset.trg), vocab_size=50000, save_path='vocab_file')

	dataset.save('dataset')
	src_field.save('src_field')
	trg_field.save('trg_field')
	src_raw_field.save('src_raw_field')


	src_examples = []
	counter = 0
	for x in dataset.src:
		src_examples.append(x)
		counter+=1
		if counter>=12: break
	
	print(src_field.process(src_examples))


def test13():
	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()
	src_raw_field = RawField()

	fields = [('src', src_field), ('trg', trg_field), ('src_raw', src_raw_field)]

	dataset = Dataset()
	dataset.load(dataset_file='dataset', fields=fields)

	src_field.load('src_field')
	trg_field.load('trg_field')
	src_raw_field.load('src_raw_field')

	src_raw_examples = []
	counter = 0
	for x in dataset.src_raw:
		src_raw_examples.append(x)
		counter+=1
		if counter>=2: break
		print('*'*5)
	
	print(src_raw_field.process(src_raw_examples))



def test14():

	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()
	src_raw_field = RawField()

	dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src,), vocab_size=50000, save_path='vocab_file')
	trg_field.build_vocab((dataset.trg,), vocab_size=50000, save_path='vocab_file')


	minibatch = []
	counter = 0
	for x in dataset:
		minibatch.append(x)
		counter+=1
		if counter>=2: break


	batch = Batch(minibatch, dataset)
	print(batch.src_raw)
	print(len(batch))


def test15():

	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()
	src_raw_field = RawField()

	dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src, dataset.trg), vocab_size=50000, save_path='vocab_file')
	trg_field.build_vocab((dataset.trg,), vocab_size=50000, save_path='vocab_file')

	batch_iterator = BucketIterator(dataset=dataset, batch_size=20, sort=False, shuffle=True, 
									sort_within_batch=True,  sort_key=lambda x: len(x.src), repeat=True)

	for batch in batch_iterator:

		# batch = batch.sort('trg')

		src, src_len = batch.src
		trg, trg_len = batch.trg
		print('batch', batch)

		print('src_len', src_len)
		print('trg_len', trg_len)


def test16():


	last_time = time()
	

	src_field = Field(add_sos=True, add_eos=True)
	trg_field = Field()
	src_raw_field = RawField()

	dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
	src_field.build_vocab((dataset.src, dataset.trg), vocab_size=50000)
	trg_field.build_vocab((dataset.trg,), vocab_size=50000)

	# bi = torchtext.BucketIterator(dataset, batch_size=100, sort_key=lambda x: len(x.src), sort=False, shuffle=True, repeat=True)
	bi = BucketIterator(dataset, batch_size=100, sort_key=lambda x: len(x.src), sort=False, shuffle=True, repeat=True, num_buckets=10)

	bg = bi.__iter__()

	for iter in range(0, 1000):
		batch = next(bg)

		# print('bi: ', bi.epoch)
		# print('epoch_iter: ', bi.epoch_iterations)

		# batch = batch.sort('trg')

		src, src_len = batch.src
		trg, trg_len = batch.trg

		# print('src_len', src_len, '   trg_len', trg_len)
		# print('src_len', len(src), '   trg_len', len(trg))

	cal_time('BucketIterator')

	'''
	for batch in bi:

		print('bi: ', bi.epoch)
		print('epoch_iter: ', bi.epoch_iterations)

		# batch = batch.sort('trg')

		src, src_len = batch.src
		trg, trg_len = batch.trg

		print('src_len', src_len, '   trg_len', trg_len)
	'''

test16()		
