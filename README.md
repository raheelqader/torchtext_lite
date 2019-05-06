# torchtext lite
A light version of torchtext library


## Sample usage
```
from torchtext_lite import *

src_path = './source.txt'
trg_path = './target.txt' 
src_max_length=100
trg_max_length=100

src_field = Field(add_sos=True, add_eos=True)
trg_field = Field()
src_raw_field = RawField()

dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
src_field.build_vocab((dataset.src, dataset.trg), vocab_size=50000)
trg_field.build_vocab((dataset.trg,), vocab_size=50000)

bi = BucketIterator(dataset, batch_size=100, sort_key=lambda x: len(x.src), repeat=True, num_buckets=10)

for batch in bi:

	print('bi: ', bi.epoch)
	print('epoch_iter: ', bi.epoch_iterations)

	src, src_len = batch.src
	trg, trg_len = batch.trg
```