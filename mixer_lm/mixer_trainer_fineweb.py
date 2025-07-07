from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer
import torch.nn as nn
import mlflow
import datasets
from datasets import load_dataset, load_from_disk

from mixer_multiconv import MultiHeadedMixer
from mixer_autoencoder import AutoencodingMixer, MemoryMixer, ProjMemoryMixer
from memory_transformer import MemoryTransformer

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

class DoubleMixerBlock(nn.Module):

	def __init__(self, dim, length, clm_mask=False, expand_conv=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernormf = nn.LayerNorm(dim)
		self.seq_layernormr = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.convf = nn.Conv1d(length, length, 1)
			self.convr = nn.Conv1d(length, length, 1)
		self.clm_mask = clm_mask
		self.expand_conv = expand_conv
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x: torch.tensor, y: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
			y = rearrange(y, 'b p t f -> (b p) t f')

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
				# # softmax weights
				# self.conv.weight.data = self.softmax(self.conv.weight.data)
				mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
				self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

		else:

			masked_convf = torch.tril(rearrange(self.convf.weight, 'f d p -> p f d'))
			self.convf.weight.data = rearrange(masked_convf, 'p f d -> f d p').contiguous()

			masked_convr = torch.triu(rearrange(self.convr.weight, 'f d p -> p f d'), diagonal=2)
			self.convr.weight.data = rearrange(masked_convr, 'p f d -> f d p').contiguous()

		residualf, residualr = x, y
		x, y = self.seq_layernormf(x), self.seq_layernormr(y)
		x, y = self.convf(x) + residualf, self.convr(y) + residualr
		residualf, residualr = x, y
		x, y = self.patch_layernorm(x), self.patch_layernorm(y)
		x, y = self.patch_ff(x) + residualf, self.patch_ff(y) + residualr
		return x, y

class MixerBlock(nn.Module):

	def __init__(self, dim, length=512, expand_conv=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, 1, padding='same')
		self.expand_conv = expand_conv

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
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
			masked_conv = torch.tril(rearrange(self.conv.weight, 'f d p -> p f d'))
			self.conv.weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

		residual = x
		x = self.seq_layernorm(x)
		x = self.conv(x) + residual
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x


class MixerHead(nn.Module):

	def __init__(self, dim, length, hidden_dim, n_heads):
		super().__init__()
		self.n_heads = n_heads
		self.proj_head = nn.ModuleList(
			[nn.Linear(dim, hidden_dim)
			for i in range(n_heads)]
			)

		self.convs = nn.ModuleList(
			[nn.Conv1d(length, length, 1)
			for i in range(n_heads)]
			)

		self.out_proj = nn.Linear(dim*n_heads, dim)
		self.softmax = nn.Softmax(dim=-1)
		self.GeLU = nn.GELU()

	def forward(self, x: torch.tensor):

		for i in range(len(self.convs)):
			masked_conv = self.softmax(torch.tril(rearrange(self.convs[i].weight, 'f d p -> p f d')))
			self.convs[i].weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

		hidden_layer = []

		for head in range(self.n_heads):
			projection = self.proj_head[i](x)
			conv_projection = self.convs[i](x)
			hidden_layer.append(conv_projection)

		# concatenate and project multi-headed output
		hidden_layer = torch.cat(hidden_layer, dim=2)
		hidden_layer = self.out_proj(hidden_layer)
		return hidden_layer

class MLPMixerBlock(nn.Module):

	def __init__(self, dim, length, **kwargs):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.conv = nn.Linear(length, length)


	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		self.conv.weight.data = torch.tril(self.conv.weight)
		residual = x
		x = self.seq_layernorm(x)
		x = rearrange(x, 'b t h -> b h t')
		x = self.conv(x)
		x = rearrange(x, 'b h t -> b t h')
		x += residual

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
				expand_conv=False
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		if tie_weights:
			self.wte.weight = self.lm_head.weight
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for i, block in enumerate(self.mixerblocks):
			x = block(x)
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output


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

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('Vocab size: ', n_vocab)

tokenized_length = 1024
dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# mixer model initialization
#model = MultiHeadedMixer(n_vocab, dim, 16, length=tokenized_length, heads=32).float().to(device)
#model = LanguageMixer(n_vocab, dim, 1).float().to(device)
#model = AutoencodingMixer(n_vocab, dim, 8, tokenized_length).float()
#model = MemoryMixer(n_vocab, dim//4, dim - dim//16, 16, tokenized_length, combination_dim='embedding').float()
model = MemoryTransformer(n_vocab, dim//2, dim-dim//8, 16, tokenized_length, combination_dim='embedding').float()

count_parameters(model)
train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c1024"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c1024"

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

# if you have a new dataset, map before loading from disk
#map_dataset(train_path, test_path)
datasets.config.IN_MEMORY_MAX_SIZE = 50e9
train_dataset = load_from_disk(train_path, keep_in_memory=None)
test_dataset = load_from_disk(test_path, keep_in_memory=None)
#print (len(train_dataset), len(test_dataset))
mlflow.end_run()
#print (train_dataset[0])
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=8000,
	gradient_accumulation_steps=1,
	learning_rate=2e-4,
	fp16=True,
	eval_strategy='steps',
	output_dir='~/Desktop/fineweb_ememory_transformer_e256c4_d512_n16_c1024_b15',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True,
	max_steps=200000
)
print ('training begun')

trainer = transformers.Trainer(
	model=model.to(device),
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

#trainer.train("/home/bbadger/Desktop/finemath_autoencoding_mixer_1024_n8_b16_c1024/checkpoint-8000")
trainer.train()
