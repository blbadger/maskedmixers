import os
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
import prettytable
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
import datasets

device = 0 if torch.cuda.is_available else 'cpu'

dim = 512
context_length = 32
llama_config_kwargs = {
	'hidden_size': dim,
	'intermediate_size': 4*dim,
	'num_hidden_layers': 16,
	'num_attention_heads': 4,
	'vocab_size': 8000
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float()

class MTPTransformer(nn.Module):

	def __init__(self, model, n_tokens=2):
		super().__init__()
		self.model = model
		self.n_tokens = n_tokens
		self.cel = torch.nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None, **kwargs):
		x = input_ids
		for i in range(self.n_tokens):
			output = self.model.lm_head(self.model.model(x)[0])
			output = rearrange(output, 'b t e -> b e t')
			shift_logits = output[..., :-(1 + i)].contiguous()
			shift_labels = labels[..., (1 + i):].contiguous()
			if 'loss' in vars():
				loss += self.cel(shift_logits, shift_labels)
			else:
				loss = self.cel(shift_logits, shift_labels)
			x = torch.argmax(output, dim=-2)

		return loss, output


model = MTPTransformer(model, n_tokens=2)
# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)


train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

#map_dataset(train_path, test_path)
datasets.config.IN_MEMORY_MAX_SIZE = 35e9
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)


mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=3,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=16,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4, 
	fp16=True, 
	evaluation_strategy='steps',
	output_dir='~/Desktop/mtp_fineweb_llama_512_n16_c512',
	optim='adamw_torch',
	overwrite_output_dir=True,
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
#trainer.train()
trainer.train('/home/bbadger/Desktop/mtp_fineweb_llama_512_n16_c512/checkpoint-152000')
