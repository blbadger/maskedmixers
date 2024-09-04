from prettytable import PrettyTable

import torch
from einops import rearrange
import transformers
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import LlamaForCausalLM


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
		self.softmax = nn.Softmax(dim=-2)		

	def forward(self, x: torch.tensor):

		for i in range(len(self.convs)):
			masked_conv = torch.tril(self.softmax(torch.tril(rearrange(self.convs[i].weight, 'f d p -> p f d'))))
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

	def __init__(self, dim, length):
		super().__init__()
		self.patch_layernorm = nn.LayerNorm(dim)
		self.seq_layernorm = nn.LayerNorm(dim)
		self.dim = dim
		self.length = length
		self.mixerhead = MixerHead(dim, length, 1024, 2)
		self.patch_ff = FeedForward(dim)
	
	def forward(self, x: torch.tensor):
		if x.dim() > 3:
			x = rearrange(x, 'b p t f -> (b p) t f')
		residual = x
		x = self.seq_layernorm(x)
		x = self.mixerhead(x) + residual

		residual = x
		x = self.patch_layernorm(x)
		x = self.patch_ff(x) + residual
		return x


class LanguageMixer(nn.Module):

	def __init__(self, n_vocab, dim, depth):
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

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/experiments/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 8).to(device)

# cached dataset
train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

def debatch_input(input_data):
	output = []
	for i in range(len(input_data)):
		if input_data[i].dim() > 1:
			input_data[i] = input_data[i].unsqueeze(1)
			output += list(input_data[i])
	return output

def batch_tokenize_input(train_text, test_text, length=2000, batch_size=4096):
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

train_data, test_data = batch_tokenize_input(train_text, valid_text)
train_data, test_data = debatch_input(train_data), debatch_input(test_data)

mlflow.end_run()
print ('training begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=2.9,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=5e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/tinystories_mixer_1024_n8_b32_c1_h2',
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
trainer.train() # '/home/bbadger/Desktop/tinystories_mixer_128_f_n8/checkpoint-748000'
for name, param in model.named_parameters():
	print (name)

