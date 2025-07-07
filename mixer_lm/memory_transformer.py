import os
import torch
import torch.nn as nn
from einops import rearrange
import transformers
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, LlamaModel
import mlflow
from datasets import load_dataset
from mixer_autoencoder import MixerBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MemoryTransformer(nn.Module):

	def __init__(self, n_vocab, encoder_dim, dim, depth, length, compression=4, combination_dim='embedding'):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, encoder_dim)
		
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
				'hidden_size': dim,
				'intermediate_size': 4*dim,
				'num_hidden_layers': depth,
				'num_attention_heads': 4,
				'vocab_size': n_vocab
			}
			decoder_configuration = LlamaConfig(**llama_config_kwargs)
			self.decoder = LlamaModel(decoder_configuration)
			self.lm_head = nn.Linear(dim, n_vocab, bias=False)
			if encoder_dim != dim:
				self.decoder_proj = nn.Linear(encoder_dim, dim)

		elif combination_dim == 'embedding':
			llama_config_kwargs = {
				'hidden_size': dim + encoder_dim//compression,
				'intermediate_size': 4*(dim + encoder_dim//compression),
				'num_hidden_layers': depth,
				'num_attention_heads': 4,
				'vocab_size': n_vocab
			}
			decoder_configuration = LlamaConfig(**llama_config_kwargs)
			self.decoder = LlamaModel(decoder_configuration)
			self.decoder_wte = nn.Embedding(n_vocab, dim)
			self.lm_head = nn.Linear(dim + encoder_dim//compression, n_vocab, bias=False)

		print (self.decoder)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.compression = compression > 1
		if self.compression:
			self.down = nn.Linear(encoder_dim, encoder_dim//compression)
			#self.up = nn.Linear(encoder_dim//compression, encoder_dim)
		

	def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
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
		
		# feed pre-concatenated input embeddings to the transformer decoder
		x = self.decoder(inputs_embeds=x, attention_mask=attention_mask)
		output = self.lm_head(x.last_hidden_state)
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
