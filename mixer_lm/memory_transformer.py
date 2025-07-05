import os
from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
import torch.nn as nn
import mlflow
from datasets import load_dataset

class AbbreviatedLlama(nn.Module):

	def __init__(self, model, tokenized_length=1024):
		super().__init__()
		self.model = model
		self.position_ids = torch.tensor([[i for i in range(tokenized_length)]]).to(device)

	def forward(self, x, **attention_mask: torch.Tensor):
		# Matrix mult instead of embedding to prevent type incompatibility
		x = x.to(device)
		position_ids = self.position_ids.repeat(input_ids.shape[0], 1).to(device)
		position_embeddings = self.model.model.rotary_emb(x, position_ids)

		for i in range(self.model.model.layers):
			x = self.model.model.layers[i](x, position_ids=position_ids, position_embeddings=position_embeddings, attention_mask=attention_mask)[0]
		return x

class MemoryTransformer(nn.Module):

	def __init__(self, decoder, n_vocab, encoder_dim, dim, depth, length, compression=4, combination_dim='token'):
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
			llama_config_kwargs = {
				'hidden_size': dim + encoder_dim,
				'intermediate_size': 4*(dim + encoder_dim),
				'num_hidden_layers': depth,
				'num_attention_heads': 4,
				'vocab_size': n_vocab
			}
			decoder_configuration = LlamaConfig(**llama_config_kwargs)

			# Initializing a model from the llama-7b style configuration
			self.decoder = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=length)
			self.lm_head = nn.Linear(dim, n_vocab, bias=False)
			if encoder_dim != dim:
				self.decoder_proj = nn.Linear(encoder_dim, dim)

		elif combination_dim == 'embedding':
			llama_config_kwargs = {
				'hidden_size': dim + encoder_dim,
				'intermediate_size': 4*(dim + encoder_dim),
				'num_hidden_layers': depth,
				'num_attention_heads': 4,
				'vocab_size': n_vocab
			}
			decoder_configuration = LlamaConfig(**llama_config_kwargs)
			self.decoder = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=length)
			self.lm_head = nn.Linear(dim + encoder_dim, n_vocab, bias=False)

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
			encoder_embedding = self.up(encoder_embedding)

		decoder_embeds = self.decoder_wte(input_ids)
		if self.combination_dim == 'token':
			if self.decoder_proj:
				encoder_embedding = self.decoder_proj(encoder_embedding)
			x = torch.cat((encoder_embedding, decoder_embeds), dim=1) # concatenation on token dim

		elif self.combination_dim == 'embedding':
			repeat_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
			x = torch.cat((repeat_embedding, decoder_embeds), dim=2) # concatenation on hidden dim

		x = self.decoder(x, attention_mask=attention_mask)
		
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