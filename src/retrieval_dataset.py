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
from safetensors.torch import save_file

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
	embeddings = torch.tensor([])
	for i in range(0, len(input_tokens)):
		if i % 100 == 0:
			print (i)
		last_hidden_layers = gen_model(
			input_tokens[i]
		)[..., -2, :].detach().to('cpu')
		# expects the model's output to be the last hidden layer
		embeddings = torch.cat((embeddings, last_hidden_layers), dim=0)

	return embeddings


tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token

train_text, test_text = load_dataset("roneneldan/TinyStories", split="train"), load_dataset("roneneldan/TinyStories", split="train")

start, split, end = 0, 40000, 50000
train_data = batch_tokenize_input(train_text, start=start, end=split)
test_data = batch_tokenize_input(train_text, start=split, end=end)
n_vocab = len(tokenizer)

# generative model initialization
tokenized_length = 512
dim = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gen_model = LanguageMixer(n_vocab, dim, 8).float().to(device)
load_model(gen_model, '/home/bbadger/Desktop/tinystories/tinystories_mixer_1024_f_8/checkpoint-160000/model.safetensors')
gen_model.eval()
target_train = embed_input(train_data)
target_test = embed_input(test_data)

query_text = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_60k.json'))]
query_train_data = batch_tokenize_input(query_text, start=start, end=split)
query_test_data = batch_tokenize_input(query_text, start=split, end=end)
query_train, query_test = embed_input(query_train_data), embed_input(query_test_data)
dictionary = {'query_train': query_train, 'query_test': query_test, 'target_train': target_train, 'target_test': target_test}
filepath = '/home/bbadger/Desktop/retrieval_50k.safetensors'
save_file(dictionary, filepath)

def generate_retrieval_dataset(query_embeddings, target_embeddings, n_context, multiples=10):
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

			target_index = random.randint(1, n_context-1) # random index to put target embedding
			matching_target = target_embeddings[i] # target the query matches
			input[target_index] = matching_target
			labels = torch.tensor(target_index-1, dtype=torch.long) # one-element label for cross-entropy loss

			inputs.append({'input_ids': input, 'labels': labels})
	return inputs


class RetrievalDataset(torch.utils.data.Dataset):

    def __init__(self, target_embeddings, query_embeddings):
    	self.target_embeddings = target_embeddings
    	self.query_embeddings = query_embeddings

    def __getitem__(self, idx):
		input = torch.zeros((n_context, query_embeddings[0].shape[1]))
		input[0] = self.query_embeddings[idx]
		exclusive_target = self.target_embeddings[:idx] + self.target_embeddings[idx+1:]
		random_insert = random.sample(exclusive_target, k=n_context-1)
		random_insert = torch.stack(random_insert, dim=0).reshape(input[1:].shape)
		input[1:] = random_insert

		target_index = random.randint(1, n_context-1) # random index to put target embedding
		matching_target = self.target_embeddings[idx] # target the query matches
		input[target_index] = matching_target
		labels = torch.tensor(target_index-1, dtype=torch.long) # one-element label for cross-entropy loss
		return {'input_ids': input, 'labels': labels}
   
    def __len__(self):
        return len(self.encodings.input_ids)m
  

with safe_open(filepath, framework="pt", device='cpu') as f:
    target_train_embeddings, target_test_embeddings = f['target_train_embeddings'], f['target_test_embeddings']
    query_train_embeddings, query_test_embeddings = f['query_train_embeddings'], f['query_test_embeddings']


train_dataset = RetrievalDataset(target_train_embeddings, query_train_embeddings)
test_dataset = RetreivalDataset(target_test_embeddings, query_test_embeddings)

# n_context = 2000
# retrieval_train_dataset = generate_retrieval_dataset(query_train, target_train, n_context)
# retrieval_test_dataset = generate_retrieval_dataset(query_test, target_test, n_context)