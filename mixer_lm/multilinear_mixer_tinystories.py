import torch
from einops import rearrange
import transformers
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer
from datasets import load_dataset
from prettytable import PrettyTable

def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

class MobiusForward(nn.Module):

	def __init__(self, dim):
		super().__init__()
		self.numerator_lin = nn.Linear(dim, dim)
		self.denominator_lin = nn.Linear(dim, dim)
		self.numerator_const = torch.ones(dim).to(device)
		self.denominator_const = torch.ones(dim).to(device)

	def forward(self, x):
		return (self.numerator_lin(x) + self.numerator_const) / (self.denominator_lin(x) + self.denominator_const + 100)


class ParallelConvForward(nn.Module):

	def __init__(self, dim, n_parallel):
		super().__init__()
		self.convs = torch.nn.ModuleList([nn.Conv1d(dim, dim, 1).to(device) for i in range(n_parallel)])
		self.gelu = nn.GELU()

	def forward(self, x: torch.tensor):
		output = 0
		for conv in self.convs:
			output += self.gelu(conv(x))
		return output


class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, parallel_convs=4):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim) 
		self.parallelconvs = ParallelConvForward(length, parallel_convs)
		self.mixer_mask = mixer_mask
		self.mobius_ff = MobiusForward(dim)

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# TODO: try splitting activations into n slices and applying n unique convs, one per slice
		# for CLM training, apply lower triangular mask to convolution weights
		if self.mixer_mask:
			for i in range(len(self.parallelconvs.convs)):
				rearranged_shape = rearrange(self.parallelconvs.convs[i].weight, 'f d p -> f (d p)').shape
				mask = torch.tril(torch.ones(rearranged_shape)).to(device)
				applied_mask = rearrange(self.parallelconvs.convs[i].weight, 'f d p -> f (d p)') * mask
				self.parallelconvs.convs[i].weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

		residual = x
		x = self.seq_layernorm(x)
		x = self.parallelconvs(x) + residual 
		residual = x
		x = self.patch_layernorm(x)
		x = self.mobius_ff(x) + residual
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
		output = self.lm_head(x)
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output

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

def tile_inputs(input_ids, tile_overlap=100, tile_size=828):
	text_length = len(input_ids[0])
	assert text_length > tile_overlap, 'Text must be longer than overlap to tile'
	tiled_arr = []
	i = 0
	while i < text_length:
		if i + tile_size <= text_length:
			tiled_arr.append(input_ids[0][i:i+tile_size])
		else:
			# pad the last tile to the appropriate length
			tokens = input_ids[0][i:i+tile_size]
			pad_length = tile_size - len(tokens)
			tokens = torch.nn.functional.pad(tokens,
											(0, pad_length),
											 mode='constant',
											 value=tokenizer.pad_token_id)
			tiled_arr.append(tokens)
		i += tile_size - tile_overlap
	return tiled_arr

def debatch_input(input_data):
	output = []
	for i in range(len(input_data)):
		if input_data[i].dim() > 1:
			input_data[i] = input_data[i].unsqueeze(1)
			output += list(input_data[i])
	return output


def batch_tokenize_input(train_text, test_text, length=20000, batch_size=1024):
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

def tokenize_input(train_text, test_text):
	train_data, test_data = [], []
	max_length = 512

	for i in range(1000000):
		input_ids = tokenizer.encode(
			train_text[i]['text'],
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
				test_text[i]['text'],
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


def reformat_inputs(train_data, test_data):
	# reformat inputs for transformer modelz`
	for i, _ in enumerate(train_data):
		train_data[i] = train_data[i].flatten()

	for i, _ in enumerate(test_data):
		test_data[i] = test_data[i].fl

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)

tokenized_length = 512
dim = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 8).float().to(device)

# one = torch.tensor([[[1, 4, 3]]]).to(device)
# two = torch.tensor([[[1, 2, 3]]]).to(device)
# print (model(one, labels=one))
# print (model(two, labels=two))
# print (model)

count_parameters(model)

# cached dataset
train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

train_data, test_data = batch_tokenize_input(train_text, valid_text)
train_data, test_data = debatch_input(train_data), debatch_input(test_data)

mlflow.end_run()
print ('training begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	warmup_steps=500,
	eval_steps=2000,
	save_steps=2000,
	learning_rate=2e-4,
	fp16=True, 
	evaluation_strategy='steps',
	output_dir='~/Desktop/tinystories_momixer_128_f_n8',
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
trainer.train()





