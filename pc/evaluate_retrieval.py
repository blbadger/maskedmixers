from train_retrieval import RetrievalMixer, RetrievalDataset
from safetensors import safe_open
import torch
from safetensors.torch import load_model

filepath = '/path/to/embeddings.safetensors' 
with safe_open(filepath, framework="pt", device='cpu') as f:
	target_test_embeddings = f.get_tensor('target_test')
	print ('targets loaded')
	query_test_embeddings = f.get_tensor('query_test')
	print ('queries loaded')

n_context = 128
test_dataset = RetrievalDataset(target_test_embeddings, query_test_embeddings, n_context=n_context)

# initialize retrieval model
retrieval_model = RetrievalMixer(512, 8, n_context).to('cuda')
retrieval_model.eval()
load_model(retrieval_model, '/home/bbadger/Desktop/retrieval_transformer_512_c128.safetensors')
batch_size = 128
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

total_correct = 0
total = 0
with torch.no_grad():
	for batch in iter(dataloader):
		loss, output = retrieval_model(batch['input_ids'].to('cuda'), labels=batch['labels'].to('cuda'))
		selections = torch.topk(output, 1, dim=1).indices.squeeze(1)
		correct = torch.ones(len(selections), dtype=int) & (selections.to('cpu') == batch['labels'].to('cpu'))
		total_correct += torch.sum(correct)
		total += len(selections)
		print (total_correct / total)
