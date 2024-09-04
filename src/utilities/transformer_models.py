import torch
from einops import rearrange
import transformers
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import numpy as np
import random
from safetensors.torch import safe_open

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
