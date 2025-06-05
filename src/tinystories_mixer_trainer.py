import torch
from einops import rearrange
import transformers
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import LlamaForCausalLM
from utilities.mixer_models import LanguageMixer
from utilities.processors import batch_tokenize_input, debatch_input
from dotenv import load_dotenv
import os

load_dotenv()
tokenizer_path = os.getenv("TINYSTORIES_TOKENIZER_PATH")
dataset_path = os.getenv("TINYSTORIES_DATASET_PATH")

tokenizer = AutoTokenizer.from_pretrained("/path/to/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 1024
n_layers = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, 8)

# cached dataset
train_text = load_dataset(dataset_path, split="train")
valid_text = load_dataset(dataset_path, split="validation")

train_data, test_data = batch_tokenize_input(train_text, valid_text, tokenizer, n_samples=20)
train_data, test_data = debatch_input(train_data), debatch_input(test_data)

mlflow.end_run()
print ('training begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=5,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/path/to/dir',
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
trainer.train() # '/home/bbadger/Desktop/tinystories_mixer_128_f_n8/checkpoint-748000'

for name, param in model.named_parameters():
	print (name)





