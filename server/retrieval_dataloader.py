import torch
import random
from safetensors.torch import safe_open

class RetrievalDataset(torch.utils.data.Dataset):

	def __init__(self, target_embeddings, query_embeddings, n_context=512):
		self.target_embeddings = target_embeddings
		self.query_embeddings = query_embeddings.unsqueeze(1)
		self.n_context = n_context
		self.prob_weights = torch.ones(self.target_embeddings.shape[0])

	def __getitem__(self, idx):
		input = torch.zeros((self.n_context, self.query_embeddings[0].shape[1]))
		input[0] = self.query_embeddings[idx]
		self.prob_weights[idx] = 0
		indices = torch.multinomial(self.prob_weights, self.n_context-1, replacement=False)
		self.prob_weights[idx] = 1
		random_insert = self.target_embeddings[indices].reshape(input[1:].shape)
		input[1:] = random_insert

		target_index = random.randint(1, self.n_context-1) # random index to put target embedding
		matching_target = self.target_embeddings[idx] # target the query matches
		input[target_index] = matching_target
		labels = torch.tensor(target_index-1, dtype=torch.long) # one-element label for cross-entropy loss
		return {'input_ids': input, 'labels': labels}
   
	def __len__(self):
		return len(self.target_embeddings)
  
filepath = '/home/bbadger/Desktop/retrieval_50k.safetensors'
with safe_open(filepath, framework="pt", device='cpu') as f:
	target_train_embeddings, target_test_embeddings = f.get_tensor('target_train'), f.get_tensor('target_test')
	query_train_embeddings, query_test_embeddings = f.get_tensor('query_train'), f.get_tensor('query_test')


train_dataset = RetrievalDataset(target_train_embeddings, query_train_embeddings)
test_dataset = RetrievalDataset(target_test_embeddings, query_test_embeddings)

print (len(train_dataset))
for i in range(50000):
	y = train_dataset[i]