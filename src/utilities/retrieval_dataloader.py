import torch
import random
import time
from safetensors.torch import safe_open
  
class RetrievalDataset(torch.utils.data.Dataset):

	def __init__(self, target_embeddings, query_embeddings, n_context=512, pre_index=False, pre_index_epochs=100, replace=False):
		self.target_embeddings = target_embeddings
		self.query_embeddings = query_embeddings.unsqueeze(1)
		self.n_context = n_context
		self.prob_weights = torch.ones(self.target_embeddings.shape[0])
		self.allocated_input = torch.zeros((self.n_context, self.query_embeddings[0].shape[1]))
		self.pre_index = pre_index
		self.replace = replace
		if pre_index:
			self.expanded_size = len(target_embeddings) * pre_index_epochs
			self.indices = []
			for i in tqdm(range(self.expanded_size)):
				self.indices.append(torch.multinomial(self.prob_weights, self.n_context-1, replacement=replace))

	def __getitem__(self, idx):
		input = torch.zeros((self.n_context, self.query_embeddings[0].shape[1]))
		input[0] = self.query_embeddings[idx]
		self.prob_weights[idx] = 0
		if self.pre_index:
			indices = self.indices[idx]
		else:
			indices = torch.multinomial(self.prob_weights, self.n_context-1, replacement=self.replace)
			# indices = np.random.multinomial(self.n_context-1, self.prob_weights)
		self.prob_weights[idx] = 1
		input[1:] = self.target_embeddings[indices]

		target_index = random.randint(1, self.n_context-1) # random index to put target embedding
		matching_target = self.target_embeddings[idx] # target the query matches
		input[target_index] = matching_target
		labels = torch.tensor(target_index-1, dtype=torch.long) # one-element label for cross-entropy loss
		retrieval_dict = {'input_ids': input, 'labels': labels}
		return retrieval_dict

	def __len__(self):
		if self.pre_index:
			return self.expanded_size
		else:
			return min(len(self.query_embeddings), len(self.target_embeddings))


def generate_retrieval_dataset(query_embeddings, target_embeddings, n_context, multiples=1):
	inputs = []
	for m in range(multiples):
		print ('multiple: ', m)
		for i, query in enumerate(query_embeddings):
			print (query_embeddings[0].shape)
			input = torch.zeros((n_context, query_embeddings[0].shape[1]))
			input[0] = query
			exclusive_target = target_embeddings[:i] + target_embeddings[i+1:]
			random_insert = random.sample(exclusive_target, k=n_context-1)
			random_insert = torch.stack(random_insert, dim=0).reshape(input[1:].shape)
			input[1:] = random_insert

			target_index = random.randint(1, n_context-1)
			matching_target = target_embeddings[i]
			input[target_index] = matching_target
			labels = torch.tensor(target_index-1, dtype=torch.long)

			inputs.append({'input_ids': input, 'labels': labels})
	return inputs

class RetrievalIndexDataset(torch.utils.data.Dataset):

	def __init__(self, target_embeddings, query_embeddings, n_context=512):
		self.target_embeddings = target_embeddings
		self.query_embeddings = query_embeddings.unsqueeze(1)
		self.n_context = n_context
		self.length = len(query_embeddings)
		self.indices = np.arange(0, self.length)
		np.random.shuffle(self.indices)

	def __getitem__(self, idx):
		if self.n_context + self.n_context*idx >= self.length:
			np.random.shuffle(self.indices)
		input = self.indices[n_context*idx: self.n_context*idx + self.n_context]
		target_index = random.randint(1, self.n_context-1) # random position to duplicate query index (at position 0)
		input[target_index] = input[0]
		input = torch.tensor(input)
		labels = torch.tensor(target_index-1, dtype=torch.long) # one-element label for cross-entropy loss
		retrieval_dict = {'input_ids': input, 'labels': labels}
		return retrieval_dict

	def __len__(self):
		return self.length

@torch.no_grad()
def embed_input(input_tokens, gen_model):
	embeddings = []
	for i in range(0, len(input_tokens)):
		if i % 100 == 0:
			print (i)
		last_hidden_layers = gen_model(
			input_tokens[i]
		)[..., -2, :].detach().to('cpu')
		# expects the model's output to be the last hidden layer
		embeddings.append(last_hidden_layers)

	# more efficient than concatenating at each iteration
	embeddings = torch.stack(embeddings).squeeze(1)
	return embeddings

if __name__ == '__main__':
	filepath = '/home/bbadger/Desktop/retrieval_50k.safetensors'
	with safe_open(filepath, framework="pt", device='cpu') as f:
		target_train_embeddings, target_test_embeddings = f.get_tensor('target_train'), f.get_tensor('target_test')
		query_train_embeddings, query_test_embeddings = f.get_tensor('query_train'), f.get_tensor('query_test')


	train_dataset = RetrievalDataset(target_train_embeddings, query_train_embeddings)
	test_dataset = RetrievalDataset(target_test_embeddings, query_test_embeddings)
	t = time.time()
	for i in range(10000):
		y = train_dataset[i]
	print (time.time() - t)