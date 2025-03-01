import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import einops
from einops import rearrange
import transformers
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, Trainer, TrainingArguments
from transformers import TextDataset, Trainer, TrainingArguments, AutoModelWithLMHead, DataCollatorForLanguageModeling
import torch.nn as nn
import mlflow
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import sentencepiece
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoModel
from safetensors.torch import load_model, save_model, load_file
from transformers import LlamaConfig, LlamaForCausalLM
import contextlib
from datasets import load_from_disk


tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
print ('n vocab: ', n_vocab)
# barebones MLP mixer, expects an embedding on input tokens
tokenized_length = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dim = 512
llama_config_kwargs = {
    'hidden_size': dim,
    'intermediate_size': 4*dim,
    'num_hidden_layers': 8,
    'num_attention_heads': 4,
    'vocab_size': 8000,
    'use_cache': False
}

# Initializing a LLaMA model
configuration = LlamaConfig(**llama_config_kwargs)

# Initializing a model from the llama-7b style configuration
model = LlamaForCausalLM(configuration).float()
#load_model(model, '/home/bbadger/Desktop/fineweb_transfixer_512_c1024/checkpoint-108000')


train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
test_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-test-c512"

train_dataset = load_from_disk(train_path)
test_dataset = load_from_disk(test_path)


mlflow.end_run()
training_arguments = transformers.TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        eval_steps=4000,
        save_steps=4000,
        learning_rate=2e-4, 
        fp16=True, 
        evaluation_strategy='steps',
        output_dir='~/Desktop/fineweb_transfixer_512_n8_c512',
        optim='adamw_torch',
        overwrite_output_dir=True,
        max_steps=96010
)

trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.train()
trainer.train('/home/bbadger/Desktop/fineweb_transfixer_512_n8_c512/checkpoint-96000')
model.eval()

iter_data = iter(test_dataset)
with contextlib.nullcontext():
	total = 0
	for i in range(10):
		item = next(iter_data)
		
		tokens = torch.tensor(item['input_ids']).unsqueeze(0)
		label_tokens = tokens.clone().detach()
		for i in range(len(label_tokens[0])):
			if int(label_tokens[0, i]) == 1:
				label_tokens[0, i] = -100
		attention_mask =torch.tensor(item["attention_mask"]).unsqueeze(0).to(device)
		output = model.forward(input_ids=tokens.to(device), labels=label_tokens.to(device), attention_mask=attention_mask)
		print ('Given loss: ', output.loss)
		total += float(output.loss)
	print ('Average loss: ', total / 20)

for i in range(30):
	tokens = next(iter_data)

string = tokenizer.decode(tokens['input_ids'])
attention_mask =torch.tensor(tokens["attention_mask"]).unsqueeze(0)
tokens = torch.tensor(tokens['input_ids']).unsqueeze(0)
print (string)
print (tokens)
gen = False
if gen:
	# tokens = tokenizer.encode(
	# 		string,
	# 		add_special_tokens=False,
	# 		return_tensors='pt'
	# 	)
	# print (tokens)
	output = model.generate(tokens, max_new_tokens=50)
	output = tokenizer.decode(output[0])
	print (output, "\n")
label_tokens = tokens.clone().detach()
for i in range(len(label_tokens[0])):
	if int(label_tokens[0, i]) == 1:
		label_tokens[0, i] = -100
print (label_tokens)
print ('Loss: ', model(tokens.to(device), labels=label_tokens.to(device), attention_mask=attention_mask.to(device)).loss)

fout = []
for i in range(10, 1, -1):
	output = model(tokens.to(device)).logits[:, -i, :]
	output_indicies = torch.topk(output, dim=-1, k=1).indices[0]
	output_token = output_indicies[0]
	fout.append(int(output_token))
	output_word = tokenizer.decode(output_token)
	output_token = output_token.to('cpu')
	tokens[0, -i+1] = int(output_token)

print (tokenizer.decode(tokens[0]))
print (fout)
print (tokenizer.decode(fout))





