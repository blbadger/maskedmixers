import torch
from einops import rearrange
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from utilities.mixer_models import LanguageMixer
from utilities.processors import batch_tokenize_input

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token

n_vocab = len(tokenizer)
tokenized_length = 512
dim = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 1).float().to(device)
print (model)

# cached dataset
train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

train_data, test_data = batch_tokenize_input(train_text, valid_text, tokenizer)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def train_model():
	model.train()
	epochs = 2
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

train_model()










