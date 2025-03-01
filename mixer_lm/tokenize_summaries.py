import os
import torch
from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer
import torch.nn as nn
from datasets import load_dataset, load_from_disk, Dataset
import sentencepiece
import json
from safetensors.torch import save_file

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token

def tokenization(example, n_ctx=128):
    tokens = tokenizer.encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=512,
			padding=True,
			padding_side='left'
		).input_ids
    return {'input_ids': tokens}

def map_dataset(array, label='summary'):
	"""
	Map dataset to tokens. Suitable for large datasets, note that split_index is low (5k means hold out 5k rows from training)
	"""
	tokenized_array = []
	count = 0 
	for sample in array:
		tokens = tokenizer.encode_plus(
			sample,
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=512,
			padding='max_length',
			padding_side='left'
		).input_ids
		tokenized_array.append(tokens[0])
	output_dict = {label: torch.stack(tokenized_array, dim=0)}
	return output_dict

def extract_tokens(dataset, limit=400000, label='text'):
	array = []
	count = 0
	for sample in dataset:
		count += 1
		if count > limit:
			break
		array.append(sample['input_ids'])
	output_dict = {label: torch.tensor(array)}
	return output_dict


query_text = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_0_50000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_50000_100000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_100000_150000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_150000_200000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_200000_250000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_250000_300000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_300000_350000.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_350000_400000.json'))]

path = "/home/bbadger/Desktop/contrastive-finemath-lpad-400k.safetensors"
summary_dataset = map_dataset(query_text, label='summary')

text_path = "/home/bbadger/Desktop/finemath-4-tokenized-train-c512-lpad-8k"
text_dataset = load_from_disk(text_path, keep_in_memory=None)
text_dataset = extract_tokens(text_dataset, label='text')
dataset = {**text_dataset, **summary_dataset}
save_file(dataset, path)










