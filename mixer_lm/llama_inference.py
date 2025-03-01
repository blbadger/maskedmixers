import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoModel
from safetensors.torch import load_model, save_model, load_file
from transformers import LlamaConfig, LlamaForCausalLM


def FeedForward(dim, expansion_factor=4):
	inner_dim = int(dim * expansion_factor)
	return nn.Sequential(
		nn.Linear(dim, inner_dim),
		nn.GELU(),
		nn.Linear(inner_dim, dim)
	)


class MixerBlock(nn.Module):

	def __init__(self, dim, length, mixer_mask=True, expansion_factor=4, dropout=0.):
		super().__init__()
		self.layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(length)
		self.dim = dim
		self.length = length
		self.patch_ff = FeedForward(dim, expansion_factor=expansion_factor)
		self.conv = nn.Conv1d(dim, dim, 1)

		# for CLM training: mask conv weights to become upper-triangular
		if mixer_mask:
			self.conv.weight = torch.nn.Parameter(torch.triu(self.conv.weight))

	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
		x = rearrange(x, 'b t f -> b f t')
		residual = x
		x = self.conv(x) + residual
		x = self.seq_layernorm(x)
		x = rearrange(x, 'b f t -> b t f')
		residual = x
		x = self.patch_ff(x) + residual
		x = self.layernorm(x)
		return x

class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth, mixer_mask=True):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.mixerblocks = nn.ModuleList(
			[MixerBlock(
				dim = dim,
				length = tokenized_length,
				mixer_mask = mixer_mask
				)
			for i in range(depth)]
			).to(device)
		self.lm_head = nn.Linear(dim, n_vocab)
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None, training=False):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.mixerblocks:
			x = block(x)
		output = self.lm_head(x)
		if labels:
			labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		if training:
			shift_logits = output[..., :-1].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			loss = self.cel(shift_logits, shift_labels)
		else:
			loss = 0
		return loss, output

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
# model = AutoModel.from_pretrained('/home/bbadger/Desktop/tinystories_mixer/checkpoint-15000')
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

# barebones MLP mixer, expects an embedding on input tokens
tokenized_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 256
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': 8,
    # 'num_heads': 16,
    'vocab_size': 4096
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float()


load_model(model, '/home/bbadger/Desktop/tinystories/tinystories_llama_256/checkpoint-96000/model.safetensors')

def debatch_input(input_data):
	output = []
	for i in range(len(input_data)):
		if input_data[i].dim() > 1:
			input_data[i] = input_data[i].unsqueeze(1)
			output += list(input_data[i])
	return output


def batch_tokenize_input(train_text, test_text, length=2000, batch_size=1024):
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

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token

train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

for i, text in enumerate(valid_text):
	if text['text'].startswith('One day, a little boy named Tim went to play with his friend, Sam'):
		print (i)

train_data, test_data = batch_tokenize_input(train_text, valid_text)
tokens = test_data[20][..., :-50]
print (tokenizer.decode(tokens[0]))

print (model(tokens[..., -50:], labels=tokens[..., -50:]).loss)
gen = True
if gen:
	# tokens = tokenizer.encode(
	# 		string,
	# 		add_special_tokens=False,
	# 		return_tensors='pt'
	# 	)
	# print (tokens)
	output = model.generate(tokens, max_new_tokens=50)
	output = tokenizer.decode(output[0])
	print (output, "\n")

# output = model(tokens).logits
# output = torch.topk(output, dim=2, k=1).indices
# output = output.flatten()
# tokens = tokenizer.decode(output)
# print (tokens)

fout = []
for i in range(50):
	output = model(tokens).logits[:, -1, :]
	output_indicies = torch.topk(output, dim=-1, k=1).indices[0]
	output_token = output_indicies[0]
	fout.append(output_token)
	output_word = tokenizer.decode(output_token)
	output_token = output_token.to('cpu')
	tokens = torch.cat((tokens, output_token.unsqueeze(0).unsqueeze(0)), dim=-1)

print (model(tokens[..., -50:], labels=tokens[..., -50:]).loss)
print (tokenizer.decode(fout))
