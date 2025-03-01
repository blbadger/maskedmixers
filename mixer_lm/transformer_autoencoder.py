import os
import prettytable
from prettytable import PrettyTable

import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import LlamaConfig, LlamaForCausalLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AutoencodingTransformer(nn.Module):

	def __init__(self, n_vocab, dim, encoder_model, decoder_model, tokenized_length=512):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.encoder = encoder_model
		self.decoder = decoder_model
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = tokenized_length

	def forward(self, input_ids, labels=None, attention_mask=None):
		x = input_ids
		x = x.to(device).squeeze(1)
		x = self.wte(x)
		
		x = self.encoder(x)

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		encoder_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
		x = encoder_embedding

		x = self.decoder(x)

		output = self.lm_head(x)
		output = rearrange(output, 'b t e -> b e t')
		loss = self.cel(output, labels)
		return loss, output


class AbbreviatedModel(nn.Module):

	def __init__(self, model, depth=8, tokenized_length=512):
		super().__init__()
		self.model = model
		self.depth = depth
		self.position_ids = torch.tensor([[i for i in range(tokenized_length)]]).to(device)

	def forward(self, input_ids: torch.Tensor, **attention_mask: torch.Tensor):
		# Matrix mult instead of embedding to prevent type incompatibility
		x = input_ids.to(device)
		position_ids = self.position_ids.repeat(input_ids.shape[0], 1).to(device)
		position_embeddings = self.model.model.rotary_emb(x, position_ids)
		# if not attention_mask is None:
		# 	attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).half()

		for i in range(self.depth):
			x = self.model.model.layers[i](x, position_ids=position_ids, position_embeddings=position_embeddings)[0]
		return x


def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	print ()
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		params = parameter.numel()
		table.add_row([name, params])
		total_params += params
	print(table)
	print(f"Total Trainable Params: {total_params}")
	return total_params



def batch_tokenize_input(train_text, test_text, length=20000, batch_size=4096):
	train_data, test_data = [], []
	max_length = 512

	for i in range(0, length, batch_size):
		tokens = tokenizer.batch_encode_plus(
			train_text[i:i+batch_size]['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=max_length,
			padding='max_length'
		)
		# debatch train data
		for i in range(tokens.input_ids.shape[0]):
			train_data.append({'input_ids': tokens.input_ids[i, :], 'attention_mask': tokens.attention_mask[i, :]})

	for i in range(0, len(test_text), batch_size):
		tokens = tokenizer.batch_encode_plus(
			test_text[i:i+batch_size]['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=True,
			max_length=max_length,
			padding='max_length'
		)
		# debatch test data
		for i in range(tokens.input_ids.shape[0]):
			test_data.append({'input_ids': tokens.input_ids[i, :], 'attention_mask': tokens.attention_mask[i, :]})
	return train_data, test_data

def reformat_inputs(train_data, test_data):
	# reformat inputs for transformer modelz`
	for i, _ in enumerate(train_data):
		train_data[i] = train_data[i].flatten()

	for i, _ in enumerate(test_data):
		test_data[i] = test_data[i].flatten()
	return train_data, test_data

if __name__ == '__main__':
	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)
	print (tokenizer.is_fast)

	tokenized_length = 512
	dim = 128
				
	llama_config_kwargs = {
		'hidden_size': dim,
		'intermediate_size': 4*dim,
		'num_hidden_layers': 8,
		'num_attention_heads': 4,
		'vocab_size': 4096
	}

	# Initializing a LLaMA model
	configuration = LlamaConfig(**llama_config_kwargs)

	# Initializing a model from the llama-7b style configuration
	encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
	decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
	model = AutoencodingTransformer(n_vocab, dim, encoder_model, decoder_model)

	count_parameters(model)

	# cached dataset
	train_text = load_dataset("roneneldan/TinyStories", split="train")
	valid_text = load_dataset("roneneldan/TinyStories", split="validation")

	train_data, test_data = batch_tokenize_input(train_text, valid_text)
	if isinstance(model, LlamaForCausalLM):
		reformat_inputs(train_data, test_data)

	mlflow.end_run()
	print ('training begun')

	training_arguments = transformers.TrainingArguments(
		num_train_epochs=7,
		per_device_train_batch_size=32,
		per_device_eval_batch_size=32,
		warmup_steps=500,
		eval_steps=4000,
		save_steps=4000,
		learning_rate=1e-4,
		fp16=True,
		evaluation_strategy='steps',
		output_dir='~/Desktop/tinystories_autoencoding_transformer_n8_b32',
		optim='adamw_torch',
		overwrite_output_dir=True,
		save_safetensors=True
	)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_data,
		eval_dataset=test_data,
		args=training_arguments,
		data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
	)


	model.train()
	trainer.train() # '/home/bbadger/Desktop/tinystories_mixer_128_f_n8/checkpoint-748000'
	for name, param in model.named_parameters():
		print (name)

