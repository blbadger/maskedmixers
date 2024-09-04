import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import save_file
from processors import batch_tokenize_input, reformat_inputs


def save_tokens(train_text, validation_text, n_samples, batch_size, tokenizer):
	assert n_samples % batch_size == 0
	train_data, test_data = batch_tokenize_input(train_text, valid_text, tokenizer, n_samples=20, batch_size=10)

	data_dict = {
		'train_data': torch.stack(train_data, dim=0), 
		'test_data': torch.stack(test_data, dim=0)
	}

	save_file(data_dict, '/home/bbadger/Desktop/tinystories_tokens.safetensors')
	return

if __name__ == '__main__':
	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
	tokenizer.pad_token = tokenizer.eos_token
	n_vocab = len(tokenizer)
	train_text = load_dataset("roneneldan/TinyStories", split="train")
	valid_text = load_dataset("roneneldan/TinyStories", split="validation")
	n_samples, batch_size = 2000, 10
	save_tokens(train_text, valid_text, n_samples, batch_size, tokenizer)