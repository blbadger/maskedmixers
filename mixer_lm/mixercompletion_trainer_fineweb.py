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
from datasets import load_dataset, load_from_disk
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors import safe_open
from safetensors.torch import save_file
import datasets


def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)


class MixerBlock(nn.Module):

	def __init__(self, dim, length):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.conv = nn.Conv1d(length, length, 1, padding='same')

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
		masked_conv = torch.tril(rearrange(self.conv.weight, 'f d p -> p f d'))
		self.conv.weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

		residual = x
		x = self.seq_layernorm(x)
		x = self.conv(x) + residual

		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x


class AutoencodingMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, length):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.split_i = length//2
		self.encoderblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = length,
				)
			for i in range(depth)]
			).to(device)
	
		self.decoderblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = length
				)
			for i in range(depth)]
			)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = tokenized_length

	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.encoderblocks:
			x = block(x)

		x[:, self.split_i:, :] = 0 # mask 

		for block in self.decoderblocks:
			x = block(x)
		
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		labels = labels[:, self.split_i:]
		output = rearrange(output, 'b t e -> b e t')[:, :, self.split_i:]
		loss = self.cel(output, labels)
		return loss, output

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('Vocab size: ', n_vocab)

tokenized_length = 512
dim = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoencodingMixer(n_vocab, dim, 8, tokenized_length)

train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

datasets.config.IN_MEMORY_MAX_SIZE = 30e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

mlflow.end_run()
print ('Training Begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/autoencoding_mixer_1024_n16_b32',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True
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
