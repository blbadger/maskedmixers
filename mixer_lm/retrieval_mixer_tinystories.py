import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import os
import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoModel
from safetensors.torch import load_model, save_model, load_file
import json
import numpy as np
import random
from datasets import Dataset
from safetensors.torch import safe_open
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

def ConvForward(dim, expansion_factor=1):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Conv1d(dim, inner_dim, 1),
		nn.GELU(),
		nn.Conv1d(inner_dim, dim, 1)
	)

class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, expand_conv=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, 1)
		self.mixer_mask = mixer_mask
		self.expand_conv = expand_conv

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
		if self.mixer_mask:
			if self.expand_conv:
				rearranged_shape = rearrange(self.conv[0].weight, 'f d p -> f (d p)').shape
				mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				applied_mask = rearrange(self.conv[0].weight, 'f d p -> f (d p)') * mask
				self.conv[0].weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

				rearranged_shape = rearrange(self.conv[2].weight, 'f d p -> f (d p)').shape
				mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				applied_mask = rearrange(self.conv[2].weight, 'f d p -> f (d p)') * mask
				self.conv[2].weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

			else:
				rearranged_shape = rearrange(self.conv.weight, 'f d p -> f (d p)').shape
				mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
				self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

		residual = x
		x = self.seq_layernorm(x)
		x = self.conv(x) + residual
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x


class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, tie_weights=False):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			 self.wte.weight = self.lm_head.weight
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.mixerblocks:
			x = block(x)
		output = x
		return output

class BidirectionalMixerBlock(nn.Module):

	def __init__(self, dim, length):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.conv = nn.Conv1d(length, length, 1)

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		residual = x
		x = self.seq_layernorm(x)
		x = self.conv(x) + residual
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x


class RetrievalMixer(nn.Module):

	def __init__(self, dim, depth, n_samples):
		super().__init__()
		# self.query_embeddings = query_embeddings.to(device)
		# self.target_embeddings = target_embeddings.to(device)
		self.mixerblocks = nn.ModuleList(
			[BidirectionalMixerBlock(
				dim = dim,
				length = n_samples,
				)
			for i in range(depth)]
			).to(device)
		self.retrieval_head = nn.Linear(dim, 1, bias=True)
		self.cel = nn.CrossEntropyLoss()
		self.softmax = torch.nn.Softmax(dim=1)

	def forward(self, input_ids, labels=None):
		# input_ids shape: [query_emb, target_emb_1, target_emb_2,...]
		# labels have dim (input_ids-1) and are one-hot
		x = input_ids
		x = x.to(device)
		# look up tensors for each index
		for block in self.mixerblocks:
			x = block(x)
		output = self.retrieval_head(x)
		target_output = output[..., 1:, :].contiguous() # first output is from query
		labels = torch.unsqueeze(labels, 1) # or target_output = torch.squeeze(target_output, dim=-1)
		loss = self.cel(target_output, labels) # compare predicted to actual match
		return loss, target_output


class TransformerBlock(nn.Module):

	def __init__(self, dim, n_samples, n_heads):
		super().__init__()
		self.attention = nn.MultiHeadedAdttention(dim, n_heads)
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		residual = x
		x = self.seq_layernorm(x)
		x = self.attention(x) + residual
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x


class RetrievalTransformer(nn.Module):

	def __init__(self, dim, depth, n_samples):
		super().__init__()
		self.mixerblocks = nn.ModuleList(
			[TransformerBlock(
				dim = dim,
				length = n_samples,
				)
			for i in range(depth)]
			).to(device)
		self.retrieval_head = nn.Linear(dim, 1, bias=True)
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None):
		# input_ids shape: [query_emb, target_emb_1, target_emb_2,...]
		# labels have dim (input_ids-1) and are one-hot
		x = input_ids
		x = x.to(device)
		for block in self.mixerblocks:
			x = block(x)
		output = self.retrieval_head(x)
		target_output = output[..., 1:, :].contiguous() # first output is from query
		labels = torch.unsqueeze(labels, 1)
		loss = self.cel(target_output, labels) # compare predicted to actual match
		return loss, output


def debatch_input(input_data):
	output = []
	for i in range(len(input_data)):
		if input_data[i].dim() > 1:
			input_data[i] = input_data[i].unsqueeze(1)
			output += list(input_data[i])
	return output


def batch_tokenize_input(train_text, batch_size=128, start=0, end=60000):
	train_data, test_data = [], []
	max_length = 512

	for i in range(start, end, batch_size):
		if isinstance(train_text[0], dict):
			input_ids = tokenizer.batch_encode_plus(
				train_text[i:i+batch_size]['text'],
				add_special_tokens=False,
				return_tensors='pt',
				truncation=True,
				max_length=max_length,
				padding='max_length'
			).input_ids
			train_data.append(input_ids)
		else:
			input_ids = tokenizer.batch_encode_plus(
				train_text[i:i+batch_size],
				add_special_tokens=False,
				return_tensors='pt',
				truncation=True,
				max_length=max_length,
				padding='max_length'
			).input_ids
			train_data.append(input_ids)

	train_data = debatch_input(train_data)
	return train_data

@torch.no_grad()
def embed_input(input_tokens):
	embeddings = []
	for i in range(0, len(input_tokens)):
		if i % 100 == 0:
			print (i)
		last_hidden_layers = gen_model(
			input_tokens[i]
		)[..., -2, :].detach().to('cpu')
		# expects the model's output to be the last hidden layer
		embeddings.append(last_hidden_layers)

	embeddings = debatch_input(embeddings)
	return embeddings



def generate_retrieval_dataset(query_embeddings, target_embeddings, n_context, multiples=1):
	inputs = []
	for m in range(multiples):
		print ('multiple: ', m)
		for i, query in enumerate(query_embeddings):
			print (query_embeddings[0].shape)
			input = torch.zeros((n_context, query_embeddings[0].shape[1]))
			input[0] = query
			exclusive_target = target_embeddings[:i] + target_embeddings[i+1:]
			random_insert = random.sample(exclusive_target, k=n_context-1)
			random_insert = torch.stack(random_insert, dim=0).reshape(input[1:].shape)
			input[1:] = random_insert

			target_index = random.randint(1, n_context-1)
			matching_target = target_embeddings[i]
			input[target_index] = matching_target
			labels = torch.tensor(target_index-1, dtype=torch.long)

			inputs.append({'input_ids': input, 'labels': labels})
	return inputs

def in_memory_dataset():
	# for latency profiling against storage-based datasets
	train_text, test_text = load_dataset("roneneldan/TinyStories", split="train"), load_dataset("roneneldan/TinyStories", split="train")

	train_data = batch_tokenize_input(train_text, start=0, end=2000)
	test_data = batch_tokenize_input(train_text, start=2000, end=4000)

	target_train = embed_input(train_data)
	target_test = embed_input(test_data)

	query_text = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_60k.json'))]
	query_train_data = batch_tokenize_input(query_text, start=0, end=2000)
	query_test_data = batch_tokenize_input(query_text, start=2000, end=4000)
	query_train, query_test = embed_input(query_train_data), embed_input(query_test_data)

	n_context = 512
	retrieval_train_dataset = generate_retrieval_dataset(query_train, target_train, n_context)
	retrieval_test_dataset = generate_retrieval_dataset(query_test, target_test, n_context)
	return retrieval_train_dataset, retrieval_test_dataset

class RetrievalDataset(torch.utils.data.Dataset):

	def __init__(self, target_embeddings, query_embeddings, n_context=512, pre_index=False, pre_index_epochs=100, replace=False):
		self.target_embeddings = target_embeddings
		self.query_embeddings = query_embeddings.unsqueeze(1)
		self.n_context = n_context
		self.prob_weights = torch.ones(self.target_embeddings.shape[0])
		self.allocated_input = torch.zeros((self.n_context, self.query_embeddings[0].shape[1]))
		self.pre_index = pre_index
		self.replace = replace
		if pre_index:
			self.expanded_size = len(target_embeddings) * pre_index_epochs
			self.indices = []
			for i in tqdm(range(self.expanded_size)):
				self.indices.append(torch.multinomial(self.prob_weights, self.n_context-1, replacement=True))

	def __getitem__(self, idx):
		input = torch.zeros((self.n_context, self.query_embeddings[0].shape[1]))
		input[0] = self.query_embeddings[idx]
		self.prob_weights[idx] = 0
		if self.pre_index:
			indices = self.indices[idx]
		else:
			indices = torch.multinomial(self.prob_weights, self.n_context-1, replacement=self.replace)
			# indices = np.random.multinomial(self.n_context-1, self.prob_weights) 
		self.prob_weights[idx] = 1
		input[1:] = self.target_embeddings[indices]

		target_index = random.randint(1, self.n_context-1) # random index to put target embedding
		matching_target = self.target_embeddings[idx] # target the query matches
		input[target_index] = matching_target
		labels = torch.tensor(target_index-1, dtype=torch.long) # one-element label for cross-entropy loss
		retrieval_dict = {'input_ids': input, 'labels': labels}
		return retrieval_dict

	def __len__(self):
		if self.pre_index:
			return self.expanded_size
		else:
			return min(len(self.query_embeddings), len(self.target_embeddings))


class RetrievalIndexDataset(torch.utils.data.Dataset):

	def __init__(self, target_embeddings, query_embeddings, n_context=512):
		self.target_embeddings = target_embeddings
		self.query_embeddings = query_embeddings.unsqueeze(1)
		self.n_context = n_context
		self.length = len(query_embeddings)
		self.indices = np.arange(0, self.length)
		np.random.shuffle(self.indices)

	def __getitem__(self, idx):
		if self.n_context + self.n_context*idx >= self.length:
			np.random.shuffle(self.indices)
		input = self.indices[n_context*idx: self.n_context*idx + self.n_context]
		target_index = random.randint(1, self.n_context-1) # random position to duplicate query index (at position 0)
		input[target_index] = input[0]
		input = torch.tensor(input)
		labels = torch.tensor(target_index-1, dtype=torch.long) # one-element label for cross-entropy loss
		retrieval_dict = {'input_ids': input, 'labels': labels}
		return retrieval_dict

	def __len__(self):
		return self.length



if __name__ == '__main__':
	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/experiments/tiny_token_4k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)

	tokenized_length = 64
	dim = 4096

	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
	tokenizer.pad_token = tokenizer.eos_token

	n_vocab = len(tokenizer)

	filepath = '/home/bbadger/Desktop/retrieval_llama_h32_penult_200k.safetensors' 

	with safe_open(filepath, framework="pt", device='cpu') as f:
		target_train_embeddings, target_test_embeddings = f.get_tensor('target_train'), f.get_tensor('target_test')
		query_train_embeddings, query_test_embeddings = f.get_tensor('query_train'), f.get_tensor('query_test')
	target_test_embeddings = target_test_embeddings[:len(query_test_embeddings)]

	#filepath = '/home/bbadger/Desktop/retrieval_mixer_512_200_550k.safetensors'
	#with safe_open(filepath, framework="pt", device='cpu') as f:
	#	target_train_embeddings = torch.cat((target_train_embeddings, f.get_tensor('target_train')))
	#	target_test_embeddings = torch.cat((target_test_embeddings, f.get_tensor('target_test')))
	#	query_train_embeddings = torch.cat((query_train_embeddings, f.get_tensor('query_train')))
	##	query_test_embeddings = torch.cat((query_test_embeddings, f.get_tensor('query_test')))

	n_context = 32
	train_dataset = RetrievalDataset(target_train_embeddings, query_train_embeddings, n_context=n_context, replace=False)
	test_dataset = RetrievalDataset(target_test_embeddings, query_test_embeddings, n_context=n_context, replace=False)
	print (len(target_test_embeddings), len(query_test_embeddings))

	# initialize retrieval model
	retrieval_model = RetrievalMixer(512, 8, n_context)
	print ('training begun')

	training_arguments = transformers.TrainingArguments(
		num_train_epochs=200,
		per_device_train_batch_size=128,
		per_device_eval_batch_size=128,
		warmup_steps=500,
		eval_steps=4000,
		save_steps=4000,
		learning_rate=1e-4,
		fp16=True,
		evaluation_strategy='steps',
		output_dir='~/Desktop/retrieval_transformer_penult_h32_200k_c32',
		optim='adamw_torch',
		overwrite_output_dir=True,
		save_safetensors=True
	)

	trainer = transformers.Trainer(
		model=retrieval_model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments
	)

	retrieval_model.train()
	trainer.train()


	# generative model initialization
	tokenized_length = 512
	dim = 512
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	gen_model = LanguageMixer(n_vocab, dim, 8).float()
	load_model(gen_model, '/home/bbadger/Desktop/tinystories/tinystories_mixer_512_flat/checkpoint-424000/model.safetensors')
	gen_model.eval()

	  
	filepath = '/home/bbadger/Desktop/retrieval_mixer_512_200k.safetensors'
	with safe_open(filepath, framework="pt", device='cpu') as f:
		target_train_embeddings, target_test_embeddings = f.get_tensor('target_train'), f.get_tensor('target_test')
		query_train_embeddings, query_test_embeddings = f.get_tensor('query_train'), f.get_tensor('query_test')

	n_context = 32
	train_dataset = RetrievalDataset(target_train_embeddings, query_train_embeddings, n_context=n_context)
	test_dataset = RetrievalDataset(target_test_embeddings, query_test_embeddings, n_context=n_context)

	# initialize retrieval model
	retrieval_model = RetrievalMixer(512, 8, n_context)
	print ('training begun')

	training_arguments = transformers.TrainingArguments(
		num_train_epochs=100,
		per_device_train_batch_size=128,
		per_device_eval_batch_size=128,
		warmup_steps=500,
		eval_steps=4000,
		save_steps=4000,
		learning_rate=1e-4,
		fp16=True,
		evaluation_strategy='steps',
		output_dir='~/Desktop/retrieval_mixer_test',
		optim='adamw_torch',
		overwrite_output_dir=True,
		save_safetensors=True
	)

	trainer = transformers.Trainer(
		model=retrieval_model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments
	)

	retrieval_model.train()
	trainer.train()
