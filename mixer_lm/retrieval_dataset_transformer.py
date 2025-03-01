import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import os
import torch
from einops import rearrange
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import random
from safetensors.torch import save_file, load_model
from transformers import LlamaConfig, LlamaForCausalLM

def debatch_input(input_data):
	output = []
	for i in range(len(input_data)):
		if input_data[i].dim() > 1:
			input_data[i] = input_data[i].unsqueeze(1)
			output += list(input_data[i])
	return output


def batch_tokenize_input(train_text, batch_size=100, start=0, end=60000):
	train_data, test_data = [], []
	max_length = 512

	for i in range(start, end, batch_size):
		if isinstance(train_text[0], dict):
			input_ids = tokenizer.batch_encode_plus(
				train_text[i:i+batch_size]['text'],
				add_special_tokens=False,
				return_tensors='pt',
				truncation=True,
				max_length=max_length,
				padding='max_length'
			).input_ids
			train_data.append(input_ids)
		else:
			input_ids = tokenizer.batch_encode_plus(
				train_text[i:i+batch_size],
				add_special_tokens=False,
				return_tensors='pt',
				truncation=True,
				max_length=max_length,
				padding='max_length'
			).input_ids
			train_data.append(input_ids)

	train_data = debatch_input(train_data)
	return train_data

@torch.no_grad()
def transformer_embed_input(input_tokens):
	embeddings = []
	for i in range(0, len(input_tokens)):
		if i % 1000 == 0:
			print (i)
		output = gen_model(
			input_tokens[i].to(0),
			output_hidden_states=True
		)
		last_hidden_layers = output.hidden_states[-1][..., -2, :].detach().to('cpu')
		# expects the model's output to be the last hidden layer
		embeddings.append(last_hidden_layers)
		# embeddings = torch.cat((embeddings, last_hidden_layers), dim=0)

	embeddings = torch.stack(embeddings).squeeze(1)
	return embeddings


tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/experiments/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token

train_text, test_text = load_dataset("roneneldan/TinyStories", split="train"), load_dataset("roneneldan/TinyStories", split="train")

start, split, end = 0, 180000, 200000
target_train_data = batch_tokenize_input(train_text, start=start, end=split)
target_test_data = batch_tokenize_input(train_text, start=split, end=end)
n_vocab = len(tokenizer)

# generative model initialization
tokenized_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dim = 512
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': 8,
    'num_attention_heads': 32,
    'vocab_size': 4096
}

# # Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# # Initializing a model from the llama-7b style configuration
gen_model = LlamaForCausalLM(configuration).float().to(0)
load_model(gen_model, '/home/bbadger/Desktop/tinystories/tinystories_llama_512_h4_lr5/checkpoint-28000/model.safetensors')

gen_model.eval()
target_train, target_test = transformer_embed_input(target_train_data), transformer_embed_input(target_test_data)

query_text = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_60k.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_60_100k.json'))]
query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_100_200k.json'))]
# query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_200_250k.json'))]
# query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_250_300k.json'))] 
# query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_300_350k.json'))]
# query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_350_400k.json'))]
# query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_400_450k.json'))]
# query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_450_500k.json'))]
# query_text += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_500_550k.json'))]

query_train_data = batch_tokenize_input(query_text, start=start, end=split)
query_test_data = batch_tokenize_input(query_text, start=split, end=end)
for i in range(10):
	print (query_text[i], train_text[i], '\n')
query_train, query_test = transformer_embed_input(query_train_data), transformer_embed_input(query_test_data)

dictionary = {'query_train': query_train, 'query_test': query_test, 'target_train': target_train, 'target_test': target_test}
filepath = '/home/bbadger/Desktop/retrieval_llama_penult_h32_200k.safetensors'
save_file(dictionary, filepath)
