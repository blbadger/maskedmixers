import os
import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import torch.nn.functional as F
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import sentencepiece
from transformers import AutoModel, LlamaConfig, LlamaForCausalLM
from safetensors.torch import load_model, save_model, load_file, safe_open
import json
import numpy as np
import random
from datasets import Dataset, load_from_disk, load_dataset
from tqdm import tqdm
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import threading
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from mixer_autoencoder import AutoencodingMixer
from transformer_autoencoder import AbbreviatedModel, AutoencodingTransformer


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

class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, prebatched_input=True):
		super().__init__()
		self.prebatched_input = prebatched_input
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				expand_conv=False
				)
			for i in range(depth)]
			)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)


	def forward(self, input_ids, matching_index, last_indices, **kwargs):
		x = input_ids
		#print (input_ids[0, 2, :], matching_index)
		if self.prebatched_input:
			x = x.squeeze(0) # p b t -> b t
		x = x.to(device)
		x = self.wte(x)
		for i, block in enumerate(self.mixerblocks):
			x = block(x)
		model_output = x
		last_indices = [int(i) for i in last_indices[0]]
		if last_indices:
			embedding_indices = last_indices
		else:
			embedding_indices = -2
		loss = infoNCEloss(model_output, matching_index=matching_index, embedding_index=embedding_indices)
		return loss, model_output


class RetrievalTransformer(nn.Module):

	def __init__(self, model, prebatched=True):
		super().__init__()
		self.model = model.model # no lm head
		self.prebatched = prebatched

	def forward(self, input_ids, matching_index, last_indices, **kwargs):
		# LlamaModel forward pass
		if self.prebatched:
			input_ids = input_ids.squeeze(0) # p b t -> b t
		model_output = self.model(input_ids)[0]
		last_indices = [int(i) for i in last_indices[0]]
		if last_indices:
			embedding_indices = last_indices
		else:
			embedding_indices = -2
		loss = infoNCEloss(model_output, matching_index=matching_index, embedding_index=embedding_indices)
		return loss, model_output

class RetrievalAutoencoder(nn.Module):

	def __init__(self, model, prebatched=True):
		super().__init__()
		self.model = model
		self.prebatched = prebatched

	def forward(self, input_ids, matching_index, last_indices, **kwargs):
		# LlamaModel forward pass
		if self.prebatched:
			input_ids = input_ids.squeeze(0) # p b t -> b t
		x = self.model.wte(input_ids)
		for block in self.model.encoderblocks:
			x = block(x)
		model_output = x
		embedding_indices = -1
		loss = infoNCEloss(model_output, matching_index=matching_index, embedding_index=embedding_indices)
		return loss, model_output


class RetrievalAutoencoderTransformer(nn.Module):

	def __init__(self, model, prebatched=True):
		super().__init__()
		self.model = model
		self.prebatched = prebatched

	def forward(self, input_ids, matching_index, last_indices, **kwargs):
		# LlamaModel forward pass
		if self.prebatched:
			x = input_ids.squeeze(0)
		x = self.model.wte(x)
		x = self.model.encoder(x)
		model_output = x
		embedding_indices = -1
		loss = infoNCEloss(model_output, matching_index=matching_index, embedding_index=embedding_indices)
		return loss, model_output


def infoNCEloss(output, matching_index=None, embedding_index=-2):
	"""
	Implements Noise-Contrastive Loss. Assumes that there is one positive pair per batch and all 
	the rest are negative samples.

	args:
		output: torch.tensor, shape [batch, token, embedding]

	kwargs:
		matching_index: Optional[None, int], integer index of correct retrieval match
		embedding_index: Union[int, arr[int]], index or indicies of the last non-pad token
	"""
	if not isinstance(embedding_index, int):
		summary_embedding = output[0, embedding_index[0], :].unsqueeze(0)
		match_embedding = output[matching_index, embedding_index[matching_index], :]
		other_embeddings = []
		for i in range(1, matching_index):
			other_embeddings.append(output[i, embedding_index[i], :])
		for i in range(matching_index+1, len(output)):
			other_embeddings.append(output[i, embedding_index[i], :])
		nonmatch_embeddings = torch.stack(other_embeddings)

	else:
		summary_embedding = output[0, embedding_index, :].unsqueeze(0) # b t e shape
		match_embedding = output[matching_index, embedding_index, :]
		nonmatch_embeddings = torch.cat((output[1:matching_index, embedding_index, :], output[matching_index+1:, embedding_index, :]), dim=0)

	cosine_sim = torch.nn.CosineSimilarity(dim=1)
	temp = 0.02
	codists = torch.exp((1/temp)*cosine_sim(summary_embedding, match_embedding)) # temperature=0.01
#	print (matching_index, torch.topk(cosine_sim(summary_embedding, output[1:, embedding_index, :]), 1, dim=0).indices)
	nondists = torch.sum(torch.exp((1/temp)*cosine_sim(summary_embedding, nonmatch_embeddings)))
	loss = -torch.sum(torch.log(codists / (codists + nondists)))
	return loss


class RetrievalDataset(torch.utils.data.Dataset):

	def __init__(self, text_tokens, summary_tokens, batch_size=32, replace=False, right_padded=False):
		self.summary_tokens = summary_tokens
		self.text_tokens = text_tokens
		self.context_length = len(summary_tokens[0])
		self.prob_weights = torch.ones(len(summary_tokens))
		self.allocated_input = torch.zeros((batch_size, self.context_length))
		self.replace = replace
		self.batch_size = batch_size

		self.pad_token = tokenizer.encode(tokenizer.pad_token)[-1]
		self.summary_indices = []
		self.text_indices = []
		self.right_padded = right_padded
		if right_padded:
			for i in range(len(self.summary_tokens)):
				j = 0
				while j in range(len(summary_tokens[i])) and int(summary_tokens[i, j]) != self.pad_token:
					j += 1
				self.summary_indices.append(j-1)

			for i in range(len(self.text_tokens)):
				j = 0
				while j in range(len(text_tokens[i])) and int(text_tokens[i, j]) != self.pad_token:
					j += 1
				self.text_indices.append(j-1)


	def __getitem__(self, idx):
		input = torch.zeros((self.batch_size, self.context_length)) # b t shape
		input[0] = self.summary_tokens[idx]
		self.prob_weights[idx] = 0
		indices = torch.multinomial(self.prob_weights, self.batch_size-1, replacement=self.replace)
		self.prob_weights[idx] = 1
		input[1:] = self.text_tokens[indices]
		target_index = random.randint(1, self.batch_size-1) # random index to put target embedding
		matching_target = self.text_tokens[idx] # target the query matches
		input[target_index] = matching_target
		labels = torch.tensor(target_index, dtype=torch.long)
		last_indices = []
		if self.right_padded:
			for i in range(len(input)):
				j = 0
				while j in range(len(input[i])) and int(input[i, j]) != self.pad_token:
					j += 1
				last_indices.append(j-2)

		retrieval_dict = {'input_ids': input.to(torch.long), 'matching_index': labels, 'last_indices': last_indices} # results in p b t shape upon load
		return retrieval_dict

	def __len__(self):
		return len(self.summary_tokens)

if __name__ == '__main__':
	# random inits different for each GPU
	local_rank = threading.get_ident() % 1231
	print (local_rank)
	torch.manual_seed(local_rank)
	random.seed(local_rank) 
	torch.cuda.manual_seed(local_rank)

	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)

	tokenized_length = 512
	dim = 512
	n_layers = 8
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	n_context = tokenized_length

	use_mixer = False
	autoencoder = True
	if use_mixer:
		#initialize retrieval model
		if use_autoencoder:
			model = AutoencodingMixer(n_vocab, dim, n_layers, n_context) 
			load_model(model, '/home/bbadger/Desktop/finemath_autoencoding_mixer_1024_n8_b32_lpad/checkpoint-500000/model.safetensors')
			retrieval_model = RetrievalAutoencoder(model)
		else:
			retrieval_model = LanguageMixer(n_vocab, dim, n_layers, n_context)
			load_model(retrieval_model, '/home/bbadger/Desktop/fineweb_mixer_512_n16_b64_c512_lpad/checkpoint-200000/model.safetensors')
			#load_model(retrieval_model, '/home/bbadger/Desktop/finemath_mixer_1024_n16_c512_lpad/checkpoint-500000/model.safetensors')

	else:
		vocab_size = 8000 # expects fineweb_tokenizer_8k
		llama_config_kwargs = {
			'hidden_size': dim,	
			'intermediate_size': 4*dim,
			'num_hidden_layers': n_layers,
			'num_attention_heads': 4,
			'vocab_size': vocab_size
		}

		# Initializing a LLaMA model
		configuration = LlamaConfig(**llama_config_kwargs)

		if autoencoder:
			encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
			decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
			model = AutoencodingTransformer(vocab_size, dim, encoder_model, decoder_model, tokenized_length=tokenized_length)
			load_model(model, '/home/bbadger/Desktop/finemath_llama_autoencoder_512_n8/checkpoint-200000/model.safetensors')
			retrieval_model = RetrievalAutoencoderTransformer(model)

		else:
			model = LlamaForCausalLM(configuration)
			load_model(model, '/home/bbadger/Desktop/finemath_llama_n16_h4_lpad_c512/checkpoint-200000/model.safetensors')
			retrieval_model = RetrievalTransformer(model).float()

	model = retrieval_model

	#path = "/home/bbadger/Desktop/contrastive-finemath-lpad-200k.safetensors"
	path = "/home/bbadger/Desktop/contrastive-finemath-lpad-400k.safetensors"
	#path = "/home/bbadger/Desktop/contrastive-finemath-rpad-200k.safetensors"
	tokens = {}
	with safe_open(path, framework="pt", device='cpu') as f:
		for k in f.keys():
			tokens[k] = f.get_tensor(k)

	split_index = 380000
	train_dataset = RetrievalDataset(tokens['text'][:split_index], tokens['summary'][:split_index], right_padded=False)
	test_dataset = RetrievalDataset(tokens['text'][split_index:], tokens['summary'][split_index:], right_padded=False)


	pad_token = int(tokenizer.encode(tokenizer.pad_token)[-1])
	training_arguments = transformers.TrainingArguments(
		num_train_epochs=1,
		per_device_train_batch_size=1, # actually defined in dataset subclass
		per_device_eval_batch_size=1, # actually defined in dataset subclass
		warmup_steps=500,
		eval_steps=10000,
		save_steps=10000,
		learning_rate=1e-4,
		fp16=True,
		evaluation_strategy='steps',
		output_dir='~/Desktop/contrastive_finemath_autoencoding_mixer_500pre_400k_1024_n8_b32',
		optim='adamw_torch',
		overwrite_output_dir=True,
		save_safetensors=True,
		logging_steps=500
	)

	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		args=training_arguments
	)

	trainer.train()
	#trainer.train("/home/bbadger/Desktop/contrastive_finemath_mixer_500k_1024_n16_b32/checkpoint-95000")
