import os
from prettytable import PrettyTable
import torch
import torch.nn as nn
from einops import rearrange
import transformers
import mlflow
from transformers import AutoTokenizer
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)

class MixerHead(nn.Module):

	def __init__(self, dim, length, hidden_dim, n_heads):
		super().__init__()
		self.n_heads = n_heads
		self.proj_head = nn.ModuleList(
			[nn.Linear(dim, hidden_dim)
			for i in range(n_heads)]
			).to(device)

		self.convs = nn.ModuleList(
			[nn.Conv1d(length, length, 1)
			for i in range(n_heads)]
			)

		self.out_proj = nn.Linear(dim*n_heads, dim)
		self.softmax = nn.Softmax(dim=-1)		

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

class MixerBlock(nn.Module):

	def __init__(self, dim, length, causal=True):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		# self.mixerhead = MixerHead(1024, 512, 512, 2)
		self.patch_ff = FeedForward(dim)
		self.conv = nn.Conv1d(length, length, 1, padding='same')
		self.causal = causal

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		if self.causal:
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

	def __init__(self, n_vocab, dim, depth, length, compression=1, double_tokens=False):
		super().__init__()
		self.double_tokens = double_tokens
		if double_tokens:
			self.wte = nn.Linear(n_vocab, dim)
			self.n_vocab = n_vocab
		else:
			self.wte = nn.Embedding(n_vocab, dim)
			
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
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(dim, dim//compression)
			self.up = nn.Linear(dim//compression, dim)

	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids
		x = x.to(device)
		if self.double_tokens:
			x_pairs = x.reshape(x.shape[0], x.shape[1]//2, 2)
			# makes a two hot tensor
			inputs = torch.nn.functional.one_hot(x_pairs[:, :, 0], self.n_vocab) + torch.nn.functional.one_hot(x_pairs[:, :, 1], self.n_vocab)

		x = self.wte(x)
		for block in self.encoderblocks:
			x = block(x)

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			encoder_embedding = self.up(encoder_embedding)

		encoder_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
		x = encoder_embedding

		for block in self.decoderblocks:
			x = block(x)
		
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
			if self.double_tokens:
				labels = labels.reshape(labels.shape[0], labels.shape[1]//2, 2)

		output = rearrange(output, 'b t e -> b e t')
		loss = self.cel(output, labels)
		return loss, output

class AutoencodingTrixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, length, compression=1, double_tokens=False):
		super().__init__()
		self.double_tokens = double_tokens
		if double_tokens:
			self.wte = nn.Linear(n_vocab, dim)
			self.n_vocab = n_vocab
		else:
			self.wte = nn.Embedding(n_vocab, dim)
			
		self.encoderblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = length,
				causal=True
				)
			for i in range(depth)]
			).to(device)
	
		llama_config_kwargs = {
			'hidden_size': dim,
			'intermediate_size': 4*dim,
			'num_hidden_layers': depth,
			'num_attention_heads': 4,
			'vocab_size': n_vocab
		}
		decoder_configuration = LlamaConfig(**llama_config_kwargs)
		self.decoder = LlamaModel(decoder_configuration)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(dim, dim//compression)
			self.up = nn.Linear(dim//compression, dim)

	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids
		x = x.to(device)
		if self.double_tokens:
			x_pairs = x.reshape(x.shape[0], x.shape[1]//2, 2)
			# makes a two hot tensor
			inputs = torch.nn.functional.one_hot(x_pairs[:, :, 0], self.n_vocab) + torch.nn.functional.one_hot(x_pairs[:, :, 1], self.n_vocab)

		x = self.wte(x)
		for block in self.encoderblocks:
			x = block(x)

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			encoder_embedding = self.up(encoder_embedding)

		encoder_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
		x = encoder_embedding
		x = self.decoder(x)
		
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
			if self.double_tokens:
				labels = labels.reshape(labels.shape[0], labels.shape[1]//2, 2)

		output = rearrange(output, 'b t e -> b e t')
		loss = self.cel(output, labels)
		return loss, output


class MemoryMixer(nn.Module):

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, compression=4, combination_dim='token'):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, encoder_dim)
		self.decoder_wte = nn.Embedding(n_vocab, dim)
		self.encoderblocks = nn.ModuleList(
				[MixerBlock(
					dim = encoder_dim,
					length = length,
					causal=False
					)
				for i in range(depth)]
			).to(device)

		self.decoder_proj = None
		self.combination_dim = combination_dim
		if combination_dim == 'token':
			self.decoderblocks = nn.ModuleList(
					[MixerBlock(
						dim = dim,
						length = length+1,
						causal=True
						)
					for i in range(depth)]
				).to(device)
			self.lm_head = nn.Linear(dim, n_vocab, bias=False)
			if encoder_dim != dim:
				self.decoder_proj = nn.Linear(encoder_dim, dim)

		elif combination_dim == 'embedding':
			self.decoderblocks = nn.ModuleList(
					[MixerBlock(
						dim = dim + encoder_dim//compression,
						length = length,
						causal=True
						)
					for i in range(depth)]
				).to(device)
			self.lm_head = nn.Linear(dim + encoder_dim//compression, n_vocab, bias=False)

		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(encoder_dim, encoder_dim//compression)
			self.up = nn.Linear(encoder_dim//compression, encoder_dim)
		

	def forward(self, input_ids, labels=None, **kwargs):
		input_ids = input_ids.to(device)
		wte_embeds = self.wte(input_ids)
		x = wte_embeds
		for block in self.encoderblocks:
			x = block(x)

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			#encoder_embedding = self.up(encoder_embedding)

		decoder_embeds = self.decoder_wte(input_ids)
		if self.combination_dim == 'token':
			if self.decoder_proj:
				encoder_embedding = self.decoder_proj(encoder_embedding)
			x = torch.cat((encoder_embedding, decoder_embeds), dim=1) # concatenation on token dim

		elif self.combination_dim == 'embedding':
			repeat_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
			x = torch.cat((repeat_embedding, decoder_embeds), dim=2) # concatenation on hidden dim

		for block in self.decoderblocks:
			x = block(x)
		
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_labels, shift_logits = labels, output
		if self.combination_dim == 'token':
			shift_logits = output[..., 1:-1].contiguous() # first 'token' is encoding
		else:
			shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous() 
		loss = self.cel(shift_logits, shift_labels)
		return loss, output


class ProjMemoryMixer(nn.Module):

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, compression=4):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, encoder_dim)
		self.decoder_wte = nn.Embedding(n_vocab, dim)
		self.encoderblocks = nn.ModuleList(
				[MixerBlock(
					dim = encoder_dim,
					length = length,
					causal=False
					)
				for i in range(depth)]
			).to(device)
	
		self.decoderblocks = nn.ModuleList(
				[MixerBlock(
					dim = dim,
					length = length,
					causal=True
					)
				for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(encoder_dim, encoder_dim//compression)
			self.up = nn.Linear(encoder_dim//compression, encoder_dim)

		self.decoder_proj = nn.Linear(encoder_dim, dim)

	def forward(self, input_ids, labels=None, **kwargs):
		input_ids = input_ids.to(device)
		wte_embeds = self.wte(input_ids)
		x = wte_embeds
		for block in self.encoderblocks:
			x = block(x)

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		if self.compression:
			encoder_embedding = self.down(encoder_embedding)
			encoder_embedding = self.up(encoder_embedding)

		if self.decoder_proj:
			encoder_embedding = self.decoder_proj(encoder_embedding)
		repeated_embeddings = encoder_embedding.repeat(1, self.tokenized_length, 1)

		decoder_embeds = self.decoder_wte(input_ids)
		x = decoder_embeds + repeated_embeddings # linear combination of h and token wtes

		for block in self.decoderblocks:
			x = block(x)
		
		output = self.lm_head(x)
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_labels, shift_logits = labels, output
		shift_logits = output[..., 1:-1].contiguous() # first 'token' is encoding
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

def batch_tokenize_input(train_text, test_text, length=2000000, batch_size=4096):
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
		test_data[i] = test_data[i].flatten()
	return train_data, test_data


if __name__ == '__main__':
	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/experiments/tiny_token_4k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)
	print (tokenizer.is_fast)

	# cached dataset
	train_text = load_dataset("roneneldan/TinyStories", split="train")
	valid_text = load_dataset("roneneldan/TinyStories", split="validation")


	tokenized_length = 512
	dim = 1024
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = AutoencodingMixer(n_vocab, dim, 8, tokenized_length)

	train_data, test_data = batch_tokenize_input(train_text, valid_text)
	train_data, test_data = debatch_input(train_data), debatch_input(test_data)

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
		output_dir='~/Desktop/autoencoding_mixer_1024_n16_b32',
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
	trainer.train('/home/bbadger/Desktop/autoencoding_mixer_1024_n16_b32/checkpoint-60000')
	for name, param in model.named_parameters():
		print (name)

