
# torchtext lite
A light version of torchtext library


## Sample usage
```
from torchtext_lite import *

src_path = './source.txt'
trg_path = './target.txt' 
src_max_length=100
trg_max_length=100

src_field = TextField(add_sos=True, add_eos=True)
trg_field = TextField()
src_raw_field = RawField()

dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
src_field.build_vocab(dataset.src, vocab_size=50000)
trg_field.build_vocab(dataset.trg, vocab_size=50000)

bi = BucketIterator(dataset, batch_size=100, sort_key=lambda x: len(x.src), repeat=True, num_buckets=10)

for batch in bi:

	print('bi: ', bi.epoch)
	print('epoch_iter: ', bi.epoch_iterations)

	src, src_len = batch.src
	trg, trg_len = batch.trg
```

## Save and load vocabulary

```
src_field.build_vocab(dataset.src, vocab_size=50000, save_path='src_vocab')
trg_field.build_vocab(dataset.trg, vocab_size=50000, save_path='trg_vocab')
```
```
src_field.build_vocab(dataset.src, vocab_size=50000, load_path='src_vocab')
trg_field.build_vocab(dataset.trg, vocab_size=50000, load_path='trg_vocab')
```

## Save and load fields
```
src_field = TextField()
trg_field = TextField()

dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
src_field.build_vocab(dataset.src, vocab_size=50000)
trg_field.build_vocab(dataset.trg, vocab_size=50000)

src_field.save('src_field')
trg_field.save('trg_field')
```
```
src_field = TextField()
trg_field = TextField()

src_field.load('src_field')
trg_field.load('trg_field')
```

## Save and load datasets
```
dataset = TextDataset((src_field, trg_field, src_raw_field), src_path, trg_path, src_max_length, trg_max_length)
src_field.build_vocab(dataset.src, vocab_size=50000)
trg_field.build_vocab(dataset.trg, vocab_size=50000)
dataset.save('e2e_dataset')
```
```
dataset = Dataset()
dataset.load('e2e_dataset')
```