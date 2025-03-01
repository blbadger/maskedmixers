import torch
import torch.nn.functional as F
import datasets
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import json
import random
from accelerate import infer_auto_device_map
from safetensors.torch import load_model
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import threading
from einops import rearrange
from tqdm import tqdm
from safetensors.torch import load_model, save_model, load_file, safe_open
from safetensors.torch import save_file
from mixer_autoencoder import AutoencodingMixer

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
		if last_indices:
			last_indices = [int(i) for i in last_indices[0]]
		if last_indices:
			embedding_indices = last_indices
		else:
			embedding_indices = -2
		return model_output


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
		if last_indices:
			last_indices = [int(i) for i in last_indices[0]]
			embedding_indices = last_indices
		else:
			embedding_indices = -2
		return model_output

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
		return model_output


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
		return model_output


def generate_embedding_sample(query_dataset, target_dataset, index, dataset_size=20000, n_context=128, replace=False):
	prob_weights = torch.ones(dataset_size)
	input = [query_dataset[index]] # embedding of query placed in input
	prob_weights[index] = 0 # zero out probability of query's target embedding chosen randomly
	random_indices = torch.multinomial(prob_weights, n_context-1, replacement=replace)
	for i in random_indices:
		input.append(target_dataset[int(i)])
	target_index = random.randint(1, n_context-1) # random index to put target embedding
	input[target_index] = target_dataset[int(index)]
	return input, target_index

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
	summary_embedding = output[0, embedding_index, :].unsqueeze(0) # b t e shape
	match_embedding = output[matching_index, embedding_index, :]
	nonmatch_embeddings = torch.cat((output[1:matching_index, embedding_index, :], output[matching_index+1:, embedding_index, :]), dim=0)
	cosine_sim = torch.nn.CosineSimilarity(dim=1)
	temp = 0.02
	codists = torch.exp((1/temp)*cosine_sim(summary_embedding, match_embedding)) # temperature=0.01
	nondists = torch.sum(torch.exp((1/temp)*cosine_sim(summary_embedding, nonmatch_embeddings)))
	loss = -torch.sum(torch.log(codists / (codists + nondists)))
	return loss


def generate_sample(query_dataset, target_dataset, index, dataset_size=20000, start_index=180000, n_context=128, replace=False):
	prob_weights = torch.ones(dataset_size)
	input = [query_dataset[index]]
	prob_weights[index-start_index] = 0
	indices = torch.multinomial(prob_weights, n_context-1, replacement=replace)
	for i in indices:
		target_text = reverse_tokenizer.decode(target_dataset[int(i+start_index)]['input_ids'])
		input.append(str(target_text))
	target_index = random.randint(1, n_context-1) # random index to put target embedding
	input[target_index] = reverse_tokenizer.decode(target_dataset[int(index)]['input_ids'])
	return input, target_index

def generate_embeddings(output_path, index=-2):
	query_dataset, target_dataset = load_dataset()
	total_correct = 0
	total = 0
	# test dataset samples only
	start, stop = 380000, 400000
	query_embeddings = []
	token_path = "/home/bbadger/Desktop/contrastive-finemath-lpad-400k.safetensors"
	tokens = {}
	with safe_open(token_path, framework="pt", device='cpu') as f:
		for k in f.keys():
			tokens[k] = f.get_tensor(k)

	for i in tqdm(range(start, stop)):
		query = tokens['summary'][i].unsqueeze(0).to(device)
		with torch.no_grad():
			outputs = retrieval_model(query, 0, [])[..., index, :].unsqueeze(0)

			# normalize embeddings
			embeddings = F.normalize(outputs, p=2, dim=1).detach().to('cpu').flatten()
			query_embeddings.append(embeddings)
	query_embeddings = torch.stack(query_embeddings).squeeze(1)

	target_embeddings = []
	for i in tqdm(range(start, stop)):
		summary = tokens['text'][i].unsqueeze(0).to(device)
		with torch.no_grad():
			outputs = retrieval_model(summary, 0, [])[..., index, :].unsqueeze(0)

			# normalize embeddings
			embeddings = F.normalize(outputs, p=2, dim=1).detach().to('cpu').flatten()
			target_embeddings.append(embeddings)
	
	target_embeddings = torch.stack(target_embeddings).squeeze(1)
	dictionary = {'query': query_embeddings, 'target': target_embeddings}
	save_file(dictionary, output_path)
	return

def load_embeddings(path):
	with safe_open(path, framework="pt", device='cpu') as f:
		target_embeddings, query_embeddings = f.get_tensor('target'), f.get_tensor('query')
	return query_embeddings, target_embeddings

def benchmark_embeddings(path, n_context=32):
	query_dataset, target_dataset = load_embeddings(path) # test set embeddings loaded
	total_correct = 0
	total = 0
	for i in tqdm(range(len(query_dataset))):
		# No need to add instruction for retrieval documents
		embeddings, target_index = generate_embedding_sample(query_dataset, target_dataset, i, n_context=n_context)
		embeddings = torch.stack(embeddings, dim=0).to(device)

		# normalize embeddings
		with torch.no_grad():
			# assumes embeddings are pre-normalized
			scores = (embeddings[:1] @ embeddings[1:].T) * 100
			top_index = int(torch.topk(scores, 1).indices[0])
			if top_index+1 == target_index:
				total_correct += 1
			total += 1

	print (f'Top-1 accuracy: ', total_correct / total)
	print ('Top index, target index', top_index, target_index)

# random inits different for each GPU
local_rank = threading.get_ident() % 1231
torch.manual_seed(local_rank)
random.seed(local_rank) 
torch.cuda.manual_seed(local_rank)

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k", pad_id=1)
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 512
n_layers = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_context = tokenized_length
vocab_size = 8000

use_autoencoder = True
use_mixer = False
use_transformer = True


if use_mixer:
	if use_autoencoder:
		model = AutoencodingMixer(n_vocab, dim, n_layers, n_context) 
		retrieval_model = RetrievalAutoencoder(model).float().to(device)
		load_model(retrieval_model, '/home/bbadger/Desktop/contrastive_finemath_autoencoding_mixer_500pre_400k_1024_n8_b32/checkpoint-95000/model.safetensors')
	else:
		#initialize retrieval model
		retrieval_model = LanguageMixer(n_vocab, dim, n_layers, n_context).float().to(device)
		load_model(retrieval_model, '/home/bbadger/Desktop/contrastive_finemath_mixer_500k_1024_n16_b32/checkpoint-95000/model.safetensors')

elif use_transformer:
	llama_config_kwargs = {
		'hidden_size': dim,	
		'intermediate_size': 4*dim,
		'num_hidden_layers': 8,
		'num_attention_heads': 4,
		'vocab_size': vocab_size
	}

	if use_autoencoder:
		encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
		decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
		model = AutoencodingTransformer(vocab_size, dim, encoder_model, decoder_model, tokenized_length=tokenized_length)
		retrieval_model = RetrievalAutoencoderTransformer(model)
		load_model(retrieval_mode, '/home/bbadger/Desktop/contrastive_finemath_autoencoding_mixer_500pre_400k_1024_n8_b32checkpoint-126667')

	else:
		# Initializing a LLaMA model
		configuration = LlamaConfig(**llama_config_kwargs)
		model = LlamaForCausalLM(configuration)
		retrieval_model = RetrievalTransformer(model, prebatched=False).float().to(device)
		load_model(retrieval_model, '/home/bbadger/Desktop/contrastive_finemath_llama_512_n16_b32_lpad_penult_400k/checkpoint-95000/model.safetensors')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
reverse_tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.decode(tokenizer.encode(tokenizer.eos_token)[-1])
print (tokenizer.encode(tokenizer.pad_token))

def load_dataset(finemath=True, second=True):
	if not finemath:
		target_dataset = datasets.load_from_disk('/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512')
		query_dataset = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_0_50000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_50000_100000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_100000_150000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_150000_200000.json'))]

	else:
		target_dataset = datasets.load_from_disk('/home/bbadger/Desktop/finemath-4-tokenized-train-c512-lpad-8k')
		query_dataset = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_0_50000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_50000_100000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_100000_150000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_150000_200000.json'))]
		if second:
			query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_200000_250000.json'))]
			query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_250000_300000.json'))]
			query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_300000_350000.json'))]
			query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_350000_400000.json'))]

	return query_dataset, target_dataset

if __name__ == "__main__":

	path = '/home/bbadger/Desktop/finemath_transformer_autoencoder_400k.safetensors'
	generate_embeddings(path, index=-1)
	contexts = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
	for context in contexts:
		print (f'Context size: {context}')
		benchmark_embeddings(path, n_context=context)

	# query_dataset, target_dataset = load_dataset()
	# total_correct = 0
	# total = 0
	# start, stop = 380000, 400000
	# for i in tqdm(range(start, stop)):
	# 	# Each query must come with a one-sentence instruction that describes the task
	# 	n_samples = 32
	# 	queries = [
	# 		query_dataset[i]
	# 	]
	# 	# No need to add instruction for retrieval documents
	# 	samples, target_index = generate_sample(query_dataset, target_dataset, i, start_index=start, n_context=n_samples)

	# 	#samples[0] = str(queries[0])
	# 	samples[0] = query_dataset[i]
	# 	max_length = 512
	# 	# Tokenize the input texts
		
	# 	batch_dict = tokenizer.batch_encode_plus(
	# 			samples,
	# 			add_special_tokens=False,
	# 			return_tensors='pt',
	# 			truncation=True,
	# 			padding='max_length',
	# 			padding_side='left', 
	# 			max_length=max_length
	# 		).to(device)
	# 	with torch.no_grad():
	# 		outputs = retrieval_model(batch_dict.input_ids, [i], [])
	# 		embeddings = outputs[:, -2, :]
	# 		# normalize embeddings
	# 		embeddings = F.normalize(embeddings, p=2, dim=1)
	# 		scores = (embeddings[:1] @ embeddings[1:].T) * 100
	# 		top_index = int(torch.topk(scores, 1).indices[0])
	# 		total += 1
	# 		if top_index+1 == target_index:
	# 			total_correct += 1
	# 		if i % 50 == 0: 
	# 			print ('Top index, target index', top_index, target_index)
	# 			print (f'Top-1 accuracy: ', total_correct / total)

	
