import prettytable
from prettytable import PrettyTable

import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import LlamaConfig, LlamaForCausalLM
from safetensors import safe_open
from safetensors.torch import save_file
import datasets

class AutoencodingTransformer(nn.Module):

	def __init__(self, n_vocab, dim, length, encoder_model, decoder_model):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)
		self.encoder = encoder_model
		self.decoder = decoder_model
		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = length
		self.split_i = length//2 

	def forward(self, input_ids, labels=None, attention_mask=None):
		x = input_ids
		x = x.to(device).squeeze(1)
		x = self.wte(x)
		
		x = self.encoder(x)
		x[:, self.split_i:, :] = 0 # mask 

		# encoder_embedding = x[:, -1, :].unsqueeze(1) # dim=[batch, token, hidden]
		# encoder_embedding = encoder_embedding.repeat(1, self.tokenized_length, 1)
		# x = encoder_embedding

		x = self.decoder(x)

		output = self.lm_head(x)
		output = rearrange(output, 'b t e -> b e t')[:, :, self.split_i:]
		labels = labels[:, self.split_i:]
		loss = self.cel(output, labels)
		return loss, output


class AbbreviatedModel(nn.Module):

	def __init__(self, model, depth=8, tokenized_length=512):
		super().__init__()
		self.model = model
		self.depth = depth
		self.position_ids = torch.tensor([[i for i in range(tokenized_length)]]).to(device)

	def forward(self, input_ids: torch.Tensor, **attention_mask: torch.Tensor):
		# Matrix mult instead of embedding to prevent type incompatibility
		x = input_ids
		position_ids = self.position_ids.repeat(input_ids.shape[0], 1).to(device)
		# if not attention_mask is None:
		# 	attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).half()

		for i in range(self.depth):
			x = self.model.model.layers[i](x, position_ids=position_ids)[0]
		return x

tokenized_length = 512
dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('Vocab size: ', n_vocab)
		
llama_config_kwargs = {
	'hidden_size': dim,
	'intermediate_size': 4*dim,
	'num_hidden_layers': 8,
	'num_attention_heads': 4,
	'vocab_size': 4096
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
encoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
decoder_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
model = AutoencodingTransformer(n_vocab, dim, tokenized_length, encoder_model, decoder_model)
train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

datasets.config.IN_MEMORY_MAX_SIZE = 30e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)

mlflow.end_run()
print ('training begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=1e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/fineweb_llamacompletion_512_n8_c512',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True,
	max_steps=200000
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


model.train()
trainer.train() # '/home/bbadger/Desktop/tinystories_mixer_128_f_n8/checkpoint-748000'
for name, param in model.named_parameters():
	print (name)

