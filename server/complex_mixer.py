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
from transformers import TextDataset, Trainer, TrainingArguments
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import LlamaConfig, LlamaForCausalLM
import torch.nn.functional as F
import math
import numpy as np

class PhaseAmplitudeGelu(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, z):
		return F.gelu(torch.abs(z)) * torch.exp(1.j * torch.angle(z))

class ComplexLayerNorm(nn.Module):
	def __init__(self, target_dim):
		super().__init__()
		self.target_dim = target_dim
		self.epsilon = 1e-6

	def forward(self, z):
		z_mod = z.abs()
		z_arg = z.angle()
		expectation = torch.mean(z_mod)
		variance = torch.var(z_mod)
		normed_z_mod = 2 * (z_mod / expectation) / torch.sqrt(variance + self.epsilon)
		z_normed = torch.polar(normed_z_mod, z_arg)
		return z_normed


def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim).to(torch.cfloat),
		PhaseAmplitudeGelu(),
		nn.Linear(inner_dim, dim).to(torch.cfloat)
	)

def ConvForward(dim, expansion_factor=1):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Conv1d(dim, inner_dim, 1).to(torch.cfloat),
		PhaseAmplitudeGelu(),
		nn.Conv1d(inner_dim, dim, 1).to(torch.cfloat)
		)

def MobiusForward(dim):
	return nn.Sequential(
		nn.Linear(dim, dim).to(torch.cfloat)
		)

class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, expand_conv=False):
		super().__init__()
		self.patch_layernorm = ComplexLayerNorm(dim)
		self.seq_layernorm = ComplexLayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.a_layer = nn.Linear(dim, dim).to(torch.cfloat)
		self.b_layer = nn.Linear(dim, dim).to(torch.cfloat)
		self.expand_conv = expand_conv
		if self.expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, 1).to(torch.cfloat)
		
		# for CLM training, apply lower triangular mask to convolution weights
		self.mixer_mask = mixer_mask

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

	def __init__(self, tokenized_length, n_vocab, dim, depth, tie_weights=False, complex_position=True):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False).to(torch.cfloat)
		if tie_weights:
			self.lm_head.weight = self.wte.weight
		self.cel = nn.CrossEntropyLoss()
		complex_position = torch.zeros(tokenized_length, dim)
		complex_position = complex_position.to(torch.cfloat)

		# positional encoding
		scale = 1 / tokenized_length
		for i in range(tokenized_length):
			complex_position[i, :] = 0 + scale*1j * i
			# complex_position[i, :] = np.exp(2*scale*(math.pi)*i*1j)
		self.complex_position = complex_position.to(device)

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		x = x.to(torch.cfloat)

		for block in self.mixerblocks:
			# apply positional encoding
			x[..., :, :] += self.complex_position
			x = block(x)

		output = self.lm_head(x).to(torch.float)
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)

tokenized_length = 512
dim = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(tokenized_length, n_vocab, dim, 8)


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

count_parameters(model)

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


def batch_tokenize_input(train_text, test_text, length=2000000, batch_size=1024):
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
train_data, test_data = debatch_input(train_data), debatch_input(test_data)

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
	learning_rate=5e-4,
	evaluation_strategy='steps',
	output_dir='~/Desktop/tinystories_cmixer_256_f_n8',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=False
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_data,
	eval_dataset=test_data,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.train()
trainer.train()









