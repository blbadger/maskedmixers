import torch
from einops import rearrange
import transformers
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM
from prettytable import PrettyTable
import math
from safetensors import safe_open

device = 'cuda' if torch.cuda.is_available else 'cpu'

dim = 1024
llama_config_kwargs = {
	'hidden_size': dim,
	'intermediate_size': 4*dim,
	'num_hidden_layers': 1,
	'num_attention_heads': 4,
	'vocab_size': 4096
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

class PositionalEncoding(nn.Module):

	def __init__(self, d_model, max_len=512):
		super().__init__()

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).to(device)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe
		return x

class LanguageTransformer(nn.Module):

	def __init__(self, n_vocab, dim, depth, n_head=8):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim).to(device)
		self.wpe = PositionalEncoding(dim)
		self.transformerblocks = nn.ModuleList(
			[nn.TransformerDecoderLayer(dim, n_head, dim*4) for i in range(depth)]).to(device)
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.clm_mask = nn.Transformer.generate_square_subsequent_mask(512, device=device)

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x).squeeze(1)
		x = self.wpe(x)
		for block in self.transformerblocks:
			x = block(x, x, tgt_mask=self.clm_mask, memory_mask=self.clm_mask, tgt_is_causal=True, memory_is_causal=True)
		output = self.lm_head(x)
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits, shift_labels)
		return loss, output


#gpt_config = transformers.OpenAIGPTConfig(vocab_size=4096, n_positions=512, n_embd=512, n_layer=8, n_head=4)
#model = transformers.OpenAIGPTLMHeadModel(gpt_config)
model = LlamaForCausalLM(configuration)
# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/experiments/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print (tokenizer.is_fast)

# Causal mask check
model = model.to(device)
model.eval()
one = torch.tensor([[1, 2, 3]]).to(device)
two = torch.tensor([[1, 2, 3]]).to(device)

print ("Ones's output: ", model(one).logits)
print ("Two's output: ", model(two).logits)

def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		params = parameter.numel()
		table.add_row([name, params])
		total_params += params
	print(table)
	print(f"Total Trainable Params: {total_params}")
	return total_params

count_parameters(model)

train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

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


def batch_tokenize_input(train_text, test_text, length=2000000, batch_size=1024):
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
	max_length = max_token_length

	for i in range(500000):
		input_ids = tokenizer.encode(
			train_text[i]['text'],
			add_special_tokens=False,
			return_tensors='pt',
			truncation=False,
			max_length=max_length,
			padding='max_length'
		)

		if len(input_ids[0]) > max_length:
			pass
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
				pass
			else:
				test_data.append(input_ids)

	return train_data, test_data

#tensors = {}
#with safe_open("/home/bbadger/Desktop/tinystories_tokens.safetensors", framework="pt", device="cpu") as f:
#   for key in f.keys():
#       tensors[key] = f.get_tensor(key)

#train_data = list(tensors['train_data'])
#test_data = list(tensors['test_data'])


def reformat_inputs(train_data, test_data):
	# reformat inputs for transformer model
	for i, _ in enumerate(train_data):
		train_data[i] = train_data[i].flatten()

	for i, _ in enumerate(test_data):
		test_data[i] = test_data[i].flatten()
	return train_data, test_data


train_data, test_data = batch_tokenize_input(train_text, valid_text)
if isinstance(model, LlamaForCausalLM):
	reformat_inputs(train_data, test_data)


mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=20,
	per_device_train_batch_size=128,
	per_device_eval_batch_size=128,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4, 
	fp16=True, 
	eval_strategy='steps',
	output_dir='~/Desktop/tinystories_transformer_n1_d1024_b128',
	optim='adamw_torch',
	overwrite_output_dir=True,
)

trainer = transformers.Trainer(
	model=model.to(device),
	train_dataset=train_data,
	eval_dataset=test_data,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.train()
trainer.train()
