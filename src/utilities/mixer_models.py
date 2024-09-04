import torch
from einops import rearrange
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset

device = 'cuda' if torch.cuda.is_available else 'cpu'

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
		nn.Conv1d(dim, inner_dim, 1, bias=False),
		nn.GELU(),
		nn.Conv1d(inner_dim, dim, 1, bias=False)
		)

class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, expand_conv=True, kernel_dim=1):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		if expand_conv:
			self.conv = ConvForward(length)
		else:
			self.conv = nn.Conv1d(length, length, kernel_dim)
		self.mixer_mask = mixer_mask
		self.expand_conv = expand_conv

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		# for CLM training, apply lower triangular mask to convolution weights
		if self.mixer_mask:
			if self.expand_conv:
				masked_conv = torch.tril(rearrange(self.conv[0].weight, 'f d p -> p f d'))
				self.conv[0].weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()
				masked_conv = torch.tril(rearrange(self.conv[2].weight, 'f d p -> p f d'))
				self.conv[2].weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

			else:
				masked_conv = self.softmax(torch.tril(rearrange(self.conv.weight, 'f d p -> p f d')))
				self.conv.weight.data = rearrange(masked_conv, 'p f d -> f d p').contiguous()

		residual = x
		x = self.seq_layernorm(x)
		x = self.conv(x) + residual
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x

class MixerHead(nn.Module):

	def __init__(self, dim, length, hidden_dim, n_heads, softmax=False):
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
		self.use_softmax = False
		if softmax:
			self.softmax = nn.Softmax(dim=-1)
			self.use_softmax = True

	def forward(self, x: torch.tensor):

		for i in range(len(self.convs)):
			if self.use_softmax:
				masked_conv = self.softmax(torch.tril(rearrange(self.convs[i].weight, 'f d p -> p f d')))
			else:
				masked_conv = torch.tril(rearrange(self.convs[i].weight, 'f d p -> p f d'))
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

class HeadedMixerBlock(nn.Module):

	def __init__(self, dim, length, n_heads=1, kernel_dim=1, softmax=False):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.n_heads = n_heads
		if n_heads > 1:
			self.conv = MixerHead(1024, 512, 512, n_heads, softmax=softmax)
		else:
			self.conv = nn.Conv1d(length, length, kernel_dim, padding='same')
		self.patch_ff = FeedForward(dim)

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		if self.n_heads == 1:
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

class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, tokenized_length=512, tie_weights=False):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				)
			for i in range(depth)]
			).to('cuda')
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


class EmbeddingMixer(LanguageMixer):

	def __init__(self, n_vocab, dim, depth):
		super().__init__(n_vocab, dim, depth)

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.mixerblocks:
			x = block(x)
		return x


class AutoencodingMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, tokenized_length=512):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.encoderblocks = nn.ModuleList(
			[HeadedMixerBlock(
				dim = dim,
				length = tokenized_length,
				)
			for i in range(depth)]
			).to(device)
	
		self.decoderblocks = nn.ModuleList(
			[HeadedMixerBlock(
				dim = dim,
				length = tokenized_length
				)
			for i in range(depth)]
			)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = tokenized_length

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.encoderblocks:
			x = block(x)

		encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		encoder_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
		x = encoder_embedding

		for block in self.decoderblocks:
			x = block(x)
		
		output = self.lm_head(x)
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		loss = self.cel(output, labels)
		return loss, output


class MultiHeadedMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, tokenized_length=512, n_heads=2, softmax=False):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[HeadedMixerBlock(
				dim = dim,
				length = tokenized_length,
				n_heads=n_heads,
				softmax=softmax
				)
			for i in range(depth)]
			).to('cuda')
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
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



class BidirectionalMixerBlock(nn.Module):

	def __init__(self, dim, length):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim)
		self.conv = nn.Conv1d(length, length, 1)

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')

		residual = x
		x = self.seq_layernorm(x)
		x = self.conv(x) + residual
		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x


class RetrievalMixer(nn.Module):

	def __init__(self, dim, depth, n_samples):
		super().__init__()
		self.mixerblocks = nn.ModuleList(
			[BidirectionalMixerBlock(
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
		# look up tensors for each index
		for block in self.mixerblocks:
			x = block(x)
		output = self.retrieval_head(x)
		target_output = output[..., 1:, :].contiguous() # first output is from query
		labels = torch.unsqueeze(labels, 1) # or target_output = torch.squeeze(target_output, dim=-1)
		loss = self.cel(target_output, labels) # compare predicted to actual match
		return loss, target_output