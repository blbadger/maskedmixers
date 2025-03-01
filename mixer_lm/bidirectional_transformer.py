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
from mixer_multiconv import MultiHeadedMixer


class BidirectionalTransformer(nn.Module):

	def __init__(self, n_vocab, dim, forward_model, reverse_model):
		super().__init__()
		self.wte = nn.Embedding(n_vocab, dim)

		self.lm_head = nn.Linear(dim, n_vocab, bias=False)
		self.cel = nn.CrossEntropyLoss()
		self.tokenized_length = tokenized_length
		self.forward_model = forward_model
		self.reverse_model = reverse_model

	def forward(self, input_ids, labels=None, attention_mask=None):
		x = input_ids
		x = x.to(device).squeeze(1)
		x = self.wte(x)
		y = torch.flip(x.clone(), dims=[1]) # reversed in token dim
		
		forward = self.forward_model(x)
		reverse = self.reverse_model(y)
		pad = torch.zeros(x.shape[0], 1, x.shape[2]).to(device)

		reverse = torch.cat([torch.flip(reverse, dims=[1])[..., 1:, :], pad], dim=1) # right pad reverse
		forward = torch.cat([pad, forward[..., :-1, :]], dim=1) # left pad forward

		output = self.lm_head(forward + reverse)
		logits = rearrange(output, 'b t e -> b e t')
		if labels.dim() > 2:
			labels = rearrange(labels, 'b p t -> b (p t)')
		loss = self.cel(logits, labels)
		return loss, output


class AbbreviatedModel(nn.Module):

	def __init__(self, model, depth=8, tokenized_length=512):
		super().__init__()
		self.model = model
		self.depth = depth
		self.position_ids = torch.tensor([[i for i in range(tokenized_length)]])

	def forward(self, input_ids: torch.Tensor, **attention_mask: torch.Tensor):
		# Matrix mult instead of embedding to prevent type incompatibility
		x = input_ids
		position_ids = self.position_ids.repeat(input_ids.shape[0], 1).to(device)
		# if not attention_mask is None:
		# 	attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).half()

		for i in range(self.depth):
			x = self.model.model.layers[i](x, position_ids=position_ids)[0]
		return x


tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)

tokenized_length = 512
dim = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
			
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
forward_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
reverse_model = AbbreviatedModel(LlamaForCausalLM(configuration), tokenized_length=tokenized_length)
model = BidirectionalTransformer(n_vocab, dim, forward_model, reverse_model).to(device)

def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	print ()
	for name, parameter in model.named_parameters():
		if not parameter.requires_grad:
			continue
		params = parameter.numel()
		table.add_row([name, params])
		total_params += params
	print(table)
	print(f"Total Trainable Params: {total_params}")
	return total_params



train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test"
def tokenization(example):
	tokens = tokenizer.batch_encode_plus(
		example['text'],
		add_special_tokens=False,
		return_tensors='pt',
		truncation=True,
		max_length=512,
		padding='max_length',
		padding_side='left'	
        )
	return tokens

def map_dataset(train_path, test_path, split_index=50000):
	"""
	Map dataset to tokens. Suitable for large datasets, note that split_index is low (5k means hold out 5k rows from training)
	"""
	train_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).skip(split_index)
	test_text = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT", streaming=False).take(split_index)

	train_dataset = train_text.map(tokenization, batched=True)
	test_dataset = test_text.map(tokenization, batched=True)
	train_dataset.save_to_disk(train_path)
	test_dataset.save_to_disk(test_path)
	print ('datasets saved to disk')
	return

#map_dataset(train_path, test_path)
train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)
mlflow.end_run()
print ('training begun')

training_arguments = transformers.TrainingArguments(
	num_train_epochs=7,
	per_device_train_batch_size=32,
	per_device_eval_batch_size=32,
	warmup_steps=500,
	eval_steps=4000,
	save_steps=4000,
	learning_rate=1e-4,
	fp16=True,
	evaluation_strategy='steps',
	output_dir='~/Desktop/fineweb_bitransformer_512_n8_b32',
	optim='adamw_torch',
	overwrite_output_dir=True,
	save_safetensors=True
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

