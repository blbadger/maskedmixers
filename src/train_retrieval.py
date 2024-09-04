import torch
from einops import rearrange
import transformers
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import numpy as np
import random
from safetensors.torch import safe_open
from tqdm import tqdm
from utilities.mixer_models import RetrievalMixer
from utilities.processors import debatch_input
from utilities.transformer_models import RetrievalTransformer
from utilities.retrieval_dataloader import embed_input, generate_retrieval_dataset, RetrievalDataset

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def in_memory_dataset():
	'''
	For latency profiling against storage-based datasets. Do not use for training,
	will result in memory overflow for all but the smallest datasets.
	'''
	train_text, test_text = load_dataset("roneneldan/TinyStories", split="train"), load_dataset("roneneldan/TinyStories", split="train")

	train_data = batch_tokenize_input(train_text, start=0, end=2000)
	test_data = batch_tokenize_input(train_text, start=2000, end=4000)

	target_train = embed_input(train_data, gen_model)
	target_test = embed_input(test_data, gen_model)

	query_text = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_60k.json'))]
	query_train_data = batch_tokenize_input(query_text, start=0, end=2000)
	query_test_data = batch_tokenize_input(query_text, start=2000, end=4000)
	query_train, query_test = embed_input(query_train_data, gen_model), embed_input(query_test_data, gen_model)

	n_context = 512
	retrieval_train_dataset = generate_retrieval_dataset(query_train, target_train, n_context)
	retrieval_test_dataset = generate_retrieval_dataset(query_test, target_test, n_context)
	return retrieval_train_dataset, retrieval_test_dataset


filepath = '/path/to/embeddings.safetensors' 
with safe_open(filepath, framework="pt", device='cpu') as f:
	target_train_embeddings, target_test_embeddings = f.get_tensor('target_train'), f.get_tensor('target_test')
	query_train_embeddings, query_test_embeddings = f.get_tensor('query_train'), f.get_tensor('query_test')
target_test_embeddings = target_test_embeddings[:len(query_test_embeddings)]

n_context = 128
train_dataset = RetrievalDataset(target_train_embeddings, query_train_embeddings, n_context=n_context, replace=True)
test_dataset = RetrievalDataset(target_test_embeddings, query_test_embeddings, n_context=n_context, replace=True)

# initialize retrieval model
retrieval_model = RetrievalMixer(512, 8, n_context)

print ('training begun')
training_arguments = transformers.TrainingArguments(
	num_train_epochs=200,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=1e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/retrieval_mixer_1024_200k_c128',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True
)

trainer = transformers.Trainer(
	model=retrieval_model,
	train_dataset=train_dataset,
	eval_dataset=test_dataset,
	args=training_arguments
)

retrieval_model.train()
trainer.train()
