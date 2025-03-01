import os
import torch
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
import torch.nn as nn
from datasets import load_dataset, load_from_disk, Dataset
import sentencepiece
import pyarrow as pa
import shutil

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token

def packed_tokenization(example, n_ctx=32):
    tokens = tokenizer.encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			padding=False,
		).input_ids
    tokens = torch.flatten(tokens, start_dim=0)
    batch_size = len(tokens) // n_ctx #if len(tokens) % n_ctx==0 else len(tokens) // n_ctx + 1
    length = n_ctx * batch_size
    #tokens = tokenizer.pad(tokens, padding='max_length', max_length=length, padding_side='right')
    tokens = tokens[:length].reshape(batch_size, n_ctx)
    return {'input_ids': tokens}

def tokenization(example, n_ctx=512):
	tokens = tokenizer.batch_encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			padding='max_length',
			padding_side='left', 
			max_length=n_ctx
		)
	return tokens

train_path = "/home/bbadger/Desktop/finemath-4-tokenized-train-c512-lpad-8k"
test_path = "/home/bbadger/Desktop/finemath-4-tokenized-test-c512-lpad-8k"

def map_dataset(train_path, test_path, split_index=50000, packed=False):
	"""
	Map dataset to tokens. Suitable for large datasets, note that split_index is low (5k means hold out 5k rows from training)
	"""
	#train_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).skip(split_index)
	#test_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).take(split_index)
	train_text = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=16).skip(split_index)
	test_text = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train", num_proc=16).take(split_index)
		
	if packed:
		batch = False
		tokenize = packed_tokenization
	else:
		batch = True
		tokenize = tokenization

	train_dataset = train_text.map(tokenize, batched=batch)
	test_dataset = test_text.map(tokenize, batched=batch)
	train_dataset.save_to_disk(train_path)
	test_dataset.save_to_disk(test_path)
	print ('Datasets saved to disk')
	return


def debatch(example):
	batch_size = len(example['input_ids'])
	keys = list(example.keys())
	for key in keys:
		if key != 'input_ids':
			example.pop(key, None)
	debatched_inputs = [{'input_ids': tokens} for tokens in example["input_ids"][0]]
	if not debatched_inputs: return [{'input_ids': torch}]
	return pa.Table.from_pylist(debatched_inputs)


if __name__ == '__main__':
	packed=False
	map_dataset(train_path, test_path, packed=packed)
	train_dataset = load_from_disk(train_path)
	test_dataset = load_from_disk(test_path)
	if packed:
		train_dataset = train_dataset.filter(lambda x: len(x['input_ids']) > 0)
		test_dataset = test_dataset.filter(lambda x: len(x['input_ids']) > 0 )
		print (test_dataset[0]['input_ids'])
		test_dataset = test_dataset.map(debatch, batched=True, batch_size=1)
		print (test_dataset[0])
		test_dataset.save_to_disk(test_path+'-debatched')
		shutil.rmtree(test_path)
		train_dataset = train_dataset.map(debatch, batched=True, batch_size=1)
		print (train_dataset[0])
		train_dataset.save_to_disk(train_path+'-debatched')
		shutil.rmtree(train_path)
	







