import torch
import torch.nn.functional as F
import datasets
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import json
import random
from accelerate import infer_auto_device_map
from safetensors.torch import load_model, safe_open
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from tqdm import tqdm
from safetensors.torch import save_file

def generate_sample(query_dataset, target_dataset, index, dataset_size=20000, start_index=180000, n_context=128, replace=False):
	prob_weights = torch.ones(dataset_size)
	input = [query_dataset[index]]
	prob_weights[index-start_index] = 0
	indices = torch.multinomial(prob_weights, n_context-1, replacement=replace)
	for i in indices:
		target_text = reverse_tokenizer.decode(target_dataset[int(i+start_index)]['input_ids'])
		input.append(str(target_text))
	target_index = random.randint(1, n_context-1) # random index to put target embedding
	input[target_index] = reverse_tokenizer.decode(target_dataset[int(index)]['input_ids'])
	return input, target_index

def generate_embedding_sample(query_dataset, target_dataset, index, dataset_size=20000, n_context=128, replace=False):
	prob_weights = torch.ones(dataset_size)
	input = [query_dataset[index]] # embedding of query placed in input
	prob_weights[index] = 0 # zero out probability of query's target embedding chosen randomly
	random_indices = torch.multinomial(prob_weights, n_context-1, replacement=replace)
	for i in random_indices:
		input.append(target_dataset[int(i)])
	target_index = random.randint(1, n_context-1) # random index to put target embedding
	input[target_index] = target_dataset[int(index)]
	return input, target_index


def last_token_pool(last_hidden_states: Tensor,
				 attention_mask: Tensor) -> Tensor:
	left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
	if left_padding:
		return last_hidden_states[:, -1]
	else:
		sequence_lengths = attention_mask.sum(dim=1) - 1
		batch_size = last_hidden_states.shape[0]
		return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
	return f'Instruct: {task_description}\nQuery: {query}'


def load_dataset(finemath=True):
	if not finemath:
		target_dataset = datasets.load_from_disk('/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512')
		query_dataset = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_0_50000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_50000_100000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_100000_150000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/fineweb_retrieval_150000_200000.json'))]

	else:
		target_dataset = datasets.load_from_disk('/home/bbadger/Desktop/finemath-4-tokenized-train-c512-8k')
		query_dataset = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_0_50000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_50000_100000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_100000_150000.json'))]
		query_dataset += [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/finemath_retrieval_150000_200000.json'))]
	return query_dataset, target_dataset

def generate_embeddings(path, max_length=512):
	query_dataset, target_dataset = load_dataset()
	total_correct = 0
	total = 0
	# test dataset samples only
	start, stop = 180000, 200000
	query_embeddings = []
	for i in tqdm(range(start, stop)):
		# Each query must come with a one-sentence instruction that describes the task
		task = 'Given a summary of a passage, find the corresponding text.'
		queries = [
			get_detailed_instruct(task, query_dataset[i])
		]

		# Tokenize the input texts
		batch_dict = tokenizer(queries, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
		with torch.no_grad():
			outputs = model(**batch_dict)
			embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

			# normalize embeddings
			embeddings = F.normalize(embeddings, p=2, dim=1).detach().to('cpu').flatten()
			query_embeddings.append(embeddings)
	query_embeddings = torch.stack(query_embeddings).squeeze(1)

	target_embeddings = []
	for i in tqdm(range(start, stop)):
		target_text = [reverse_tokenizer.decode(target_dataset[i]['input_ids'])]
		# Tokenize the input texts
		batch_dict = tokenizer(target_text, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)
		with torch.no_grad():
			outputs = model(**batch_dict)
			embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

			# normalize embeddings
			embeddings = F.normalize(embeddings, p=2, dim=1).detach().to('cpu').flatten()
			target_embeddings.append(embeddings)
	
	target_embeddings = torch.stack(target_embeddings).squeeze(1)
	dictionary = {'query': query_embeddings, 'target': target_embeddings}
	save_file(dictionary, path)
	return


def load_embeddings(path):
	with safe_open(path, framework="pt", device='cpu') as f:
		target_embeddings, query_embeddings = f.get_tensor('target'), f.get_tensor('query')
	return query_embeddings, target_embeddings


def benchmark_embeddings(path, n_context=32):
	query_dataset, target_dataset = load_embeddings(path) # test set embeddings loaded
	total_correct = 0
	total = 0
	for i in tqdm(range(len(query_dataset))):
		# No need to add instruction for retrieval documents
		embeddings, target_index = generate_embedding_sample(query_dataset, target_dataset, i, n_context=n_context)
		embeddings = torch.stack(embeddings, dim=0).to(device)

		# normalize embeddings
		with torch.no_grad():
			# assumes embeddings are pre-normalized
			scores = (embeddings[:1] @ embeddings[1:].T) * 100
			top_index = int(torch.topk(scores, 1).indices[0])
			if top_index+1 == target_index:
				total_correct += 1
			total += 1

	print (f'Top-1 accuracy: ', total_correct / total)
	print ('Top index, target index', top_index, target_index)


def benchmark_samples():
	query_dataset, target_dataset = load_dataset()
	total_correct = 0
	total = 0
	start, stop = 180000, 200000
	for i in tqdm(range(start, stop)):
		# Each query must come with a one-sentence instruction that describes the task
		n_samples = 32
		task = 'Given a summary of a passage, find the corresponding text.'
		query = get_detailed_instruct(task, query_dataset[i])
		
		# No need to add instruction for retrieval documents
		samples, target_index = generate_sample(query_dataset, target_dataset, i, n_context=n_samples)

		#samples[0] = str(queries[0])
		samples[0] = query
		max_length = 512
		# Tokenize the input texts
		batch_dict = tokenizer(samples, max_length=max_length, padding=True, truncation=True, return_tensors='pt').to(device)	
		with torch.no_grad():
			outputs = model(**batch_dict)
			embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

			# normalize embeddings
			embeddings = F.normalize(embeddings, p=2, dim=1)
			scores = (embeddings[:1] @ embeddings[1:].T) * 100
			top_index = int(torch.topk(scores, 1).indices[0])
			if top_index+1 == target_index:
				total_correct += 1
			total += 1
			# if i % 5 == 0:
			print (f'Top-1 accuracy: ', total_correct / total)
			print ('Top index, target index', top_index, target_index)


if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	bnb_config = BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_use_double_quant=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_compute_dtype=torch.float16
	)

	# device = 'cuda' if torch.cuda.is_available() else 'cpu'
	# tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/e5-mistral-7b-instruct")
	# model = AutoModel.from_pretrained("/home/bbadger/Desktop/e5-mistral-7b-instruct", quantization_config=bnb_config, device_map='auto')
	# #model = AutoModel.from_pretrained("/home/bbadger/Desktop/e5-mistral-7b-instruct", torch_dtype=torch.float16, device_map='auto')
	# reverse_tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")

	path = '/home/bbadger/Desktop/finemath_mistral_retrieval_200k_test.safetensors'
	# generate_embeddings(path)
	benchmark_embeddings(path, n_context=8192)
