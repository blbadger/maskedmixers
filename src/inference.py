import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from einops import rearrange
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_model
from utilities.mixer_models import LanguageMixer

class LanguageMixerwLoss(LanguageMixer):

	def __init__(self, n_vocab, dim, depth, tie_weights=False, loss_window=100):
		super().__init__(n_vocab, dim, depth)
		self.loss_window = loss_window

	def forward(self, input_ids, labels=None):
		x = input_ids
		x = x.to(device)
		x = self.wte(x)
		for block in self.mixerblocks:
			x = block(x)
		output = self.lm_head(x)
		labels = rearrange(labels, 'b p t -> b (p t)')
		output = rearrange(output, 'b t e -> b e t')
		labels = labels.to(device)
		shift_logits = output[..., :-1].contiguous()
		shift_labels = labels[..., 1:].contiguous()
		loss = self.cel(shift_logits[..., -loss_window:], shift_labels[..., -loss_window:])
		print (f'Windowed CEL: {loss}') # observe loss on last 100 tokens
		return loss, output

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token

train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

n_vocab = len(tokenizer)

tokenized_length = 512
dim = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixerwLoss(n_vocab, dim, 8).float().to(device)
load_model(model, '/path/to/model.safetensors')
model.eval()
prompt = valid_text[10]['text']
tokens = tokenizer.encode(prompt, return_tensors='pt', padding='max_length', max_length=512)

tokens = tokenizer.encode(
				prompt,
				add_special_tokens=False,
				return_tensors='pt',
				padding='max_length',
				max_length=512
			)

print ('model loaded.')
print ('Input: ', tokenizer.decode(tokens[0]))
tokens = rearrange(tokens, '(b p) t -> b p t', p=1)

fout = []
for i in range(50, 1, -1):
	loss, output = model(tokens, labels=tokens.to(device))
	out_token = torch.topk(output, dim=1, k=1).indices.flatten()[-i]
	tokens[..., -i+1] = out_token

print ('\n \n')
print ('Output: \n', tokenizer.decode(tokens[0][0]))

