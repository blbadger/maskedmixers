import torch
import transformers
import mlflow
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import LlamaConfig, LlamaForCausalLM
from prettytable import PrettyTable
from safetensors.torch import save_file
from safetensors import safe_open
from utilities.processors import batch_tokenize_input, reformat_inputs

device = 'cuda' if torch.cuda.is_available else 'cpu'

def init_model(mtype='llama', dim=512, n_vocab=512):
	if mtype == 'llama':
		llama_config_kwargs = {
		    'hidden_size': dim,
		    'intermediate_size': 4*dim,
		    'num_hidden_layers': 8,
		    'num_attention_heads': 32,
		    'vocab_size': 4096
		}

		# Initializing a LLaMA model
		configuration = LlamaConfig(**llama_config_kwargs)
		model = LlamaForCausalLM(configuration).float()

	else:
		# for GPT initialization
		gpt_config = transformers.OpenAIGPTConfig(vocab_size=4096, n_positions=n_vocab, n_embd=dim, n_layer=8, n_head=4)
		model = transformers.OpenAIGPTLMHeadModel(gpt_config)
	return model


tokenizer = AutoTokenizer.from_pretrained("/path/to/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
dim = 512
model = init_model()

tensors = {}
with safe_open("/path/to/tinystories_tokens.safetensors", framework="pt", device="cpu") as f:
   for key in f.keys():
       tensors[key] = f.get_tensor(key)

train_data = list(tensors['train_data'])
test_data = list(tensors['test_data'])
if isinstance(model, LlamaForCausalLM):
	reformat_inputs(train_data, test_data)

mlflow.end_run()
training_arguments = transformers.TrainingArguments(
	num_train_epochs=20,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=2e-4, 
	fp16=True, 
	evaluation_strategy='steps',
	output_dir='~/Desktop/llama_512',
	optim='adamw_torch',
	overwrite_output_dir=True,
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

