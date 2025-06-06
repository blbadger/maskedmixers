from mixer_retrieval_tinystories import RetrievalMixer, RetrievalDataset
from safetensors import safe_open
import torch
from safetensors.torch import load_model

# embeddings path
filepath = '/home/bbadger/Desktop/finemath_mixer_1024_retrieval_200k.safetensors' 
with safe_open(filepath, framework="pt", device='cpu') as f:
	target_test_embeddings = f.get_tensor('target_test')
	print ('targets loaded')
	query_test_embeddings = f.get_tensor('query_test')
	print ('queries loaded')

n_context = 32
test_dataset = RetrievalDataset(target_test_embeddings, query_test_embeddings, n_context=n_context)

# initialize retrieval model
retrieval_model = RetrievalMixer(1024, 8, n_context).to('cuda')
retrieval_model.eval()
# retrieval model path
load_model(retrieval_model, '/home/bbadger/Desktop/finemath_retrieval_mixer_1024_n8_c32_400k/checkpoint-8000/model.safetensors')
batch_size = 128
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Tests models for correctness batch by batch: note that this is very inefficient for actual benchmarking, and should only be used for
# reference against pre-embedded measurements
total_correct = 0
total = 0
with torch.no_grad():
	for batch in iter(dataloader):
		loss, output = retrieval_model(batch['input_ids'].to('cuda'), labels=batch['labels'].to('cuda'))
		selections = torch.topk(output, 1, dim=1).indices.squeeze(1).squeeze(1)
		correct = torch.sum(torch.ones(len(selections), dtype=int) & (selections.to('cpu') == batch['labels'].to('cpu')))
		total_correct += correct
		total += len(selections)
		print (total_correct / total)

