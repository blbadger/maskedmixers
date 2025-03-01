import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
from safetensors.torch import load_model


def FeedForward(dim, expansion_factor=1):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		# nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

def ConvForward(dim, expansion_factor=1):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Conv1d(dim, inner_dim, 1),
		nn.GELU(),
		nn.Conv1d(inner_dim, dim, 1)
		)

class LinearBlock(nn.Module):

	def __init__(self, dim, length, clm_mask=True):
		super().__init__()
		self.dim = dim
		self.length = length
		self.conv = nn.Conv1d(length, length, 1)
		self.clm_mask = clm_mask

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
		if self.clm_mask:
			rearranged_shape = rearrange(self.conv.weight, 'f d p -> f (d p)').shape
			mask = torch.tril(torch.ones(rearranged_shape)).to(device)
			applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
			self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

		residual = x
		x = self.conv(x) + residual
		return x

class MixerBlock(nn.Module):

	def __init__(self, dim, length, clm_mask=True, expand_conv=False):
		super().__init__()
		# self.patch_layernorm = nn.LayerNorm(dim)
		# self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		# self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, 1)
		self.clm_mask = clm_mask
		self.expand_conv = expand_conv
		# self.softmax = nn.Softmax(dim=0)

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
		if self.clm_mask:
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
		x = self.conv(x) + residual
		return x


class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, tie_weights=False):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				clm_mask=True,
				expand_conv=True
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
		output = self.lm_head(x)
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output

class LinearMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[LinearBlock(
				dim = dim,
				length = tokenized_length,
				clm_mask=True,
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.mixerblocks:
			x = block(x)
		
		if labels:
			output = self.lm_head(x)
			labels = rearrange(labels, 'b p t -> b (p t)')
			output = rearrange(output, 'b t e -> b e t')
			shift_logits = output[..., :-1].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			loss = self.cel(shift_logits, shift_labels)
			return loss, output
		else:
			return x

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenized_length = 64

# class MixerBlock(nn.Module):

# 	def __init__(self, dim, length, clm_mask=True):
# 		super().__init__()
# 		self.conv = nn.Conv1d(length, length, 1)
# 		self.clm_mask = clm_mask

# 	def forward(self, x: torch.tensor):
# 		if x.dim() > 3:
# 			x = rearrange(x, 'b p t f -> (b p) t f')

# 		masked_conv = torch.tril(rearrange(self.conv.weight, 'f d p -> p f d'))
# 		self.conv.weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

# 		residual = x
# 		x = self.conv(x) + residual
# 		return x


# class LanguageMixer(nn.Module):

# 	def __init__(self, n_vocab, dim, depth):
# 		super().__init__()
# 		self.wte = nn.Embedding(n_vocab, dim)
# 		self.mixerblocks = nn.ModuleList(
# 			[MixerBlock(
# 				dim = dim,
# 				length = tokenized_length,
# 				clm_mask=True
# 				)
# 			for i in range(depth)]
# 			).to(device)
# 		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
# 		self.cel = nn.CrossEntropyLoss()

# 	def forward(self, input_ids, labels=None):
# 		x = input_ids
# 		x = x.to(device)
# 		x = self.wte(x)
# 		for block in self.mixerblocks:
# 			x = block(x)
# 		output = self.lm_head(x)
# 		labels = rearrange(labels, 'b p t -> b (p t)')
# 		output = rearrange(output, 'b t e -> b e t')
# 		shift_logits = output[..., :-1].contiguous()
# 		shift_labels = labels[..., 1:].contiguous()
# 		loss = self.cel(shift_logits, shift_labels)
# 		return loss, output

# # tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
# tokenizer.pad_token = tokenizer.eos_token
# n_vocab = len(tokenizer)
# print (tokenizer.is_fast)

# tokenized_length = 128
# dim = 512
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = LanguageMixer(n_vocab, dim, 1).float().to(device)

if __name__ == '__main__':

	dim = 4096
	model = LanguageMixer(n_vocab, dim, 1).float().to(device)
	load_model(model, '/home/bbadger/Desktop/linear_mixer_4096.safetensors')
	print ('Model loaded')

	# cached dataset
	train_text = load_dataset("roneneldan/TinyStories", split="train")
	valid_text = load_dataset("roneneldan/TinyStories", split="validation")

	def debatch_input(input_data):
		output = []
		for i in range(len(input_data)):
			if input_data[i].dim() > 1:
				input_data[i] = input_data[i].unsqueeze(1)
				output += list(input_data[i])
		return output


	def batch_tokenize_input(train_text, test_text, length=20000, batch_size=16384):
		train_data, test_data = [], []
		max_length = 512

		for i in range(0, length, batch_size):
			input_ids = tokenizer.batch_encode_plus(
				train_text[i:i+batch_size]['text'],
				add_special_tokens=False,
				return_tensors='pt',
				truncation=True,
				max_length=max_length,
				padding='max_length'
			).input_ids
			train_data.append(input_ids)

		for i in range(0, len(test_text), batch_size):
			input_ids = tokenizer.batch_encode_plus(
				test_text[i:i+batch_size]['text'],
				add_special_tokens=False,
				return_tensors='pt',
				truncation=True,
				max_length=max_length,
				padding='max_length'
			).input_ids
			test_data.append(input_ids)

		train_data = debatch_input(train_data)
		test_data = debatch_input(test_data)

		return train_data, test_data

	train_data, test_data = batch_tokenize_input(train_text, valid_text)
	# train_data, test_data = debatch_input(train_data), debatch_input(test_data)

	def train_solver(model, train_data):
		train_batch = train_data[0]
		train_batch = batch.to(device) 
		loss, output = model(batch) 
		loss.backward()
		minimal_params = torch.pinv(model.grad()) @ torch.zeros(train_batch.shape)
		return minimal_params

	train_solver(model, train_data)
	print (list(model.named_parameters()))





