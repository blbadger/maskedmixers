import torch
from einops import rearrange
import transformers
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import random
from transformers import LlamaConfig
from safetensors.torch import safe_open
from utilities.transformer_models import RetrievalTransformer, embed_input
from utilities.processors import batch_tokenize_input, debatch_input
from utilities.retrieval_dataloader import RetrievalDataset

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/experiments/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


filepath = '/path/to/transformer_embeddings.safetensors'
with safe_open(filepath, framework="pt", device='cpu') as f:
	target_train_embeddings, target_test_embeddings = f.get_tensor('target_train'), f.get_tensor('target_test')
	query_train_embeddings, query_test_embeddings = f.get_tensor('query_train'), f.get_tensor('query_test')

n_context = 32
train_dataset = RetrievalDataset(target_train_embeddings, query_train_embeddings, n_context=n_context)
test_dataset = RetrievalDataset(target_test_embeddings, query_test_embeddings, n_context=n_context)

# initialize retrieval model
retrieval_model = RetrievalTransformer(1024, 8, n_context) # dim to match mixer retrieval
print ('training begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=200,
	per_device_train_batch_size=128,
	per_device_eval_batch_size=128,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=1e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/retrieval_2transformers_1024_n32_200k',
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
