from prettytable import PrettyTable
import torch
from einops import rearrange
import transformers
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import LlamaForCausalLM
from mixer_models import LangaugeMixer
from processors import reformat_inputs, batch_tokenize_input, debatch_input

# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/experiments/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 2048
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 6).float()

# cached dataset
train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

train_data, test_data = batch_tokenize_input(train_text, valid_text)
train_data, test_data = debatch_input(train_data), debatch_input(test_data)

mlflow.end_run()
print ('training begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=1,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=16,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/tinystories_mixer_trainer_test',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True
)

trainer = transformers.Trainer(
	model=model,
	train_dataset=train_data,
	eval_dataset=test_data,
	args=training_arguments,
	data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.train()
trainer.train()







