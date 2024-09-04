import torch
from einops import rearrange
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_model
import json
import random
from safetensors.torch import save_file
from utilities.mixer_models import EmbeddingMixer
from utilities.processors import retrieval_tokenize
from utilities.retrieval_dataloader import embed_input

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_model(dim, mixer=True):
	if not mixer:
		llama_config_kwargs = {
		    'hidden_size': dim,
		    'intermediate_size': 4*dim,
		    'num_hidden_layers': 8,
		    'num_heads': 16,
		    'vocab_size': 4096
		}

		# Initializing a LLaMA model
		configuration = LlamaConfig(**llama_config_kwargs)

		# Initializing a model from the llama-7b style configuration
		model = LlamaForCausalLM(configuration).float()
		load_model(model, '/path/to/model.safetensors')
	else:
		gen_model = EmbeddingMixer(n_vocab, dim, 8).float().to(device) 
		load_model(gen_model, '/path/to/model.safetensors')

	return gen_model


tokenizer = AutoTokenizer.from_pretrained("/path/to/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
train_text, test_text = load_dataset("roneneldan/TinyStories", split="train"), load_dataset("roneneldan/TinyStories", split="train")

start, split, end = 0, 4000, 5000
train_data = retrieval_tokenize(train_text, tokenizer, start=start, end=split)
test_data = retrieval_tokenize(train_text, tokenizer, start=split, end=end)
n_vocab = len(tokenizer)

# generative model initialization
tokenized_length = 512
dim = 1024

gen_model = init_model(dim, mixer=True)
gen_model.eval()

target_train = embed_input(train_data, gen_model)
target_test = embed_input(test_data, gen_model)

query_text = [i['choices'][0]['message']['content'] for i in json.load(open('/path/to/summaries.json'))]
query_train_data = retrieval_tokenize(query_text, start=start, end=split)
query_test_data = retrieval_tokenize(query_text, start=split, end=end)
query_train = embed_input(query_train_data, gen_model) 
query_test = embed_input(query_test_data, gen_model)
dictionary = {'query_train': query_train, 'query_test': query_test, 'target_train': target_train, 'target_test': target_test}
filepath = '/path/to/embeddings.safetensors'
save_file(dictionary, filepath)