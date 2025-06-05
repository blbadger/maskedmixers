import torch
from einops import rearrange
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from utilities.mixer_models import LanguageMixer
from utilities.processors import batch_tokenize_input
from safetensors.torch import save_file
from dotenv import load_dotenv
import os


def train_model(model, epochs=2):
	"""
	Custom trainer for validating model gradient flow and optimizers
	"""
	model.train()
	for epoch in range(epochs):
		total_loss = 0
		for step, batch in enumerate(train_data):
			batch = rearrange(batch, 'b (p t) -> b p t', p=1)
			optimizer.zero_grad()
			batch = batch.to(device) # discard class labels
			loss, output = model(batch, batch)
			total_loss += loss.item()
			loss.backward()
			print (model.mixerblocks[0].conv[0].weight.grad[10][:10])
			optimizer.step()
			print (model.mixerblocks[0].conv[0].weight[10][:10])
		print ('Average loss: ', total_loss / len(batch))
	save_file(model, "path/to/model.safetensors")
`	return


if __name__ == '__main__'
	load_dotenv()
	tokenizer_path = os.getenv("TINYSTORIES_TOKENIZER_PATH")
	dataset_path = os.getenv("TINYSTORIES_DATASET_PATH")

	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
	tokenizer.pad_token = tokenizer.eos_token

	n_vocab = len(tokenizer)
	tokenized_length = 512
	dim = 128
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = LanguageMixer(n_vocab, dim, 1).float().to(device)
	print (model)

	# cached dataset
	train_text = load_dataset(dataset_path, split="train")
	valid_text = load_dataset(dataset_path, split="validation")

	train_data, test_data = batch_tokenize_input(train_text, valid_text, tokenizer)
	optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
	train_model(model)










