import os
import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import LlamaConfig, LlamaForCausalLM
import prettytable
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import datasets
from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer

device = 0 if torch.cuda.is_available else 'cpu'

dim = 512
context_length = 512
vocab_size = 8000
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': 8,
    'num_attention_heads': 4,
    'vocab_size': vocab_size
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
#model = LlamaForCausalLM(configuration).float()

# Initializing a model from the llama-7b style configuration
encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=context_length)
model = AutoencodingTransformer(vocab_size, dim, encoder_model, decoder_model, tokenized_length=context_length)

# cached dataset
# train_text = load_dataset("roneneldan/TinyStories", split="train")
# valid_text = load_dataset("roneneldan/TinyStories", split="validation")

# gpt_config = transformers.OpenAIGPTConfig(vocab_size=8000, n_positions=512, n_embd=512, n_layer=16, n_head=4)
# model = transformers.OpenAIGPTLMHeadModel(gpt_config)

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)
#print (model)

# Causal mask check
# model = model.to(device)
# one = torch.tensor([[1, 2, 5]]).to(device)
# two = torch.tensor([[1, 2, 3]]).to(device)
# print (model(one, labels=one).logits)
# print (model(two, labels=two).logits)
# print (model)

def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		params = parameter.numel()
		table.add_row([name, params])
		total_params += params
	print(table)
	print(f"Total Trainable Params: {total_params}")
	return total_params

count_parameters(model)

def tokenization(example):
    tokens = tokenizer.batch_encode_plus(
			example['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=128,
			padding='max_length',
			padding_side='right'
		)
    return tokens


def map_dataset(train_path, test_path, split_index=50000):
	"""
	Map dataset to tokens. Suitable for large datasets, note that split_index is low (5k means hold out 5k rows from training)
	"""
	train_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).skip(split_index)
	test_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).take(split_index)

	train_dataset = train_text.map(tokenization, batched=True)
	test_dataset = test_text.map(tokenization, batched=True)
	train_dataset.save_to_disk(train_path)
	test_dataset.save_to_disk(test_path)
	print ('datasets saved to disk')
	return

train_path = "/home/bbadger/Desktop/finemath-4-tokenized-train-c512-lpad-8k"
test_path = "/home/bbadger/Desktop/finemath-4-tokenized-test-c512-lpad-8k"

#map_dataset(train_path, test_path)
datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

def tokenize_input(train_text, test_text):
	train_data, test_data = [], []
	max_length = 512

	for i, sample in enumerate(train_text):
		if i % 10000 == 0: print (i)
		input_ids = tokenizer.encode(
			sample['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			max_length=max_length,
			padding='max_length'
		)

		if len(input_ids[0]) > max_length:
			input_set = tile_inputs(input_ids, tile_size=max_length)
			for inp in input_set:
				train_data.append(inp)
		else:
			train_data.append(input_ids)

	for i in range(len(test_text)):
		if test_text[i]:
			input_ids = tokenizer.encode(
				test_text[i],
				add_special_tokens=False,
				return_tensors='pt',
				truncation=False,
				max_length=max_length,
				padding='max_length'
			)

			if len(input_ids[0]) > max_length:
				
				input_set = tile_inputs(
					input_ids,
					tile_size=max_length
				)
				for inp in input_set:
					test_data.append(inp)
			else:
				test_data.append(input_ids)

	return train_data, test_data

tokenize=False
if tokenize:
	print ('tokenizing input')
	train_data, test_data = tokenize_input(train_text, valid_text)
	#train_data, test_data = debatch_input(train_data), debatch_input(test_data)

	data_dict = {
	'train_data': torch.stack(train_data, dim=0), 
	'test_data': torch.stack(test_data, dim=0)
	}

	save_file(data_dict, '/home/bbadger/Desktop/tokenized_fineweb10b_16k.safetensors')
	print ('tokens saved')

load_input = False
if load_input:
	tensors = {}
	with safe_open("/home/bbadger/Desktop/tokenized_fineweb10b_8k.safetensors", framework="pt", device="cpu") as f:
		for key in f.keys():
			tensors[key] = f.get_tensor(key)

	train_data = list(tensors['train_data'])
	test_data = list(tensors['test_data'])
	print (train_data[0].shape)

def reformat_inputs(train_data, test_data):
	# reformat inputs for transformer model
	for i, _ in enumerate(train_data):
		train_data[i] = train_data[i].flatten()

	for i, _ in enumerate(test_data):
		test_data[i] = test_data[i].flatten()
	return train_data, test_data


mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4, 
	fp16=True, 
	evaluation_strategy='steps',
	output_dir='~/Desktop/finemath_llama_autoencoder_512_n8',
	optim='adamw_torch',
	overwrite_output_dir=True,
	max_steps=200000
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.train()
trainer.train() 
#trainer.train('/home/bbadger/Desktop/finemath_llama_n16_h4_lpad_c512/checkpoint-76000')
