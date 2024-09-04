import torch
from einops import rearrange
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_model
from transformers import LlamaConfig, LlamaForCausalLM
from utilities.processors import batch_tokenize_input 

tokenizer = AutoTokenizer.from_pretrained("/path/to/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 128
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

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token

train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

train_data, test_data = batch_tokenize_input(train_text, valid_text, tokenizer)
tokens = test_data[20][..., :-50]
print (tokenizer.decode(tokens[0]))
print (model(tokens[..., -50:], labels=tokens[..., -50:]).loss)

gen = True
if gen:
	output = model.generate(tokens, max_new_tokens=50)
	output = tokenizer.decode(output[0])
	print (output, "\n")

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