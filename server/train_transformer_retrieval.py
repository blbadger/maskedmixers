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


class TransformerBlock(nn.Module):

        def __init__(self, dim, n_samples, n_heads=4):
                super().__init__()
                self.attention = nn.MultiheadAttention(dim, n_heads)
                self.patch_layernorm = nn.LayerNorm(dim)
                self.seq_layernorm = nn.LayerNorm(dim)
                self.dim = dim 
                self.patch_ff = FeedForward(dim)

        def forward(self, x: torch.tensor):
                if x.dim() > 3:
                        x = rearrange(x, 'b p t f -> (b p) t f')

                residual = x 
                x = self.seq_layernorm(x)
                key, query, value = self.key_proj(x), self.query_proj(x), self.value_proj(x)
                x = self.attention(x) + residual
                residual = x 
                x = self.patch_layernorm(x)
                x = self.patch_ff(x) + residual
                return x

class RetrievalTransformer(nn.Module):

	def __init__(self, dim, depth, n_samples, n_head=4):
		super().__init__()
		self.mixerblocks = nn.ModuleList(
		[TransformerBlock(
			dim,
			n_samples,
		)
		for i in range(depth)]
		).to(device)
		self.transformerblocks = nn.ModuleList(
		[nn.TransformerDecoderLayer(dim, n_head, dim*4) for i in range(depth)]
		).to(device)
		self.retrieval_head = nn.Linear(dim, 1, bias=True)
		self.cel = nn.CrossEntropyLoss()

	def forward(self, input_ids, labels=None):
		# input_ids shape: [query_emb, target_emb_1, target_emb_2,...]
		# labels have dim (input_ids-1) and are one-hot
		x = input_ids
		x = x.to(device)
		for block in self.transformerblocks:
			x = block(x, x)
		output = self.retrieval_head(x)
		target_output = output[..., 1:, :].contiguous() # first output is from query
		labels = torch.unsqueeze(labels, 1)
		loss = self.cel(target_output, labels) # compare predicted to actual match
		return loss, output

@torch.no_grad()
def embed_input(input_tokens):
	embeddings = []
	for i in range(0, len(input_tokens)):
		if i % 100 == 0:
			print (i)
		output = gen_model(
			input_tokens[i].to(0),
			output_hidden_states=True
		)
		last_hidden_layers = output.hidden_states[-1][..., -1, :].detach().to('cpu')
		# expects the model's output to be the last hidden layer
		embeddings.append(last_hidden_layers)

	embeddings = debatch_input(embeddings)
	return embeddings

tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/experiments/tiny_token_4k")
tokenizer.pad_token = tokenizer.eos_token
n_vocab = len(tokenizer)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def in_memory_dataset():
	# for latency profiling against storage-based datasets
	train_text, test_text = load_dataset("roneneldan/TinyStories", split="train"), load_dataset("roneneldan/TinyStories", split="train")

	train_data = batch_tokenize_input(train_text, start=0, end=2000)
	test_data = batch_tokenize_input(train_text, start=2000, end=4000)

	target_train = embed_input(train_data)
	target_test = embed_input(test_data)

	query_text = [i['choices'][0]['message']['content'] for i in json.load(open('/home/bbadger/Desktop/train_output_60k.json'))]
	query_train_data = batch_tokenize_input(query_text, start=0, end=2000)
	query_test_data = batch_tokenize_input(query_text, start=2000, end=4000)
	query_train, query_test = embed_input(query_train_data), embed_input(query_test_data)

	n_context = 512
	retrieval_train_dataset = generate_retrieval_dataset(query_train, target_train, n_context)
	retrieval_test_dataset = generate_retrieval_dataset(query_test, target_test, n_context)
	return retrieval_train_dataset, retrieval_test_dataset

class RetrievalDataset(torch.utils.data.Dataset):

	def __init__(self, target_embeddings, query_embeddings, n_context=512, pre_index=False):
		self.target_embeddings = target_embeddings
		self.query_embeddings = query_embeddings.unsqueeze(1)
		self.n_context = n_context
		self.prob_weights = torch.ones(self.target_embeddings.shape[0])
		self.allocated_input = torch.zeros((self.n_context, self.query_embeddings[0].shape[1]))
		self.indices = None
		if pre_index:
			self.indices = [torch.multinomial(self.prob_weights, self.n_context-1, replacement=False) for i in range(len(target_embeddings))]

	def __getitem__(self, idx):
		input = self.allocated_input
		input[0] = self.query_embeddings[idx]
		self.prob_weights[idx] = 0
		if self.indices:
			indices = self.indices[idx]
		else:
			indices = torch.multinomial(self.prob_weights, self.n_context-1, replacement=False) 
		self.prob_weights[idx] = 1
		input[1:] = self.target_embeddings[indices]

		target_index = random.randint(1, self.n_context-1) # random index to put target embedding
		matching_target = self.target_embeddings[idx] # target the query matches
		input[target_index] = matching_target
		labels = torch.tensor(target_index-1, dtype=torch.long) # one-element label for cross-entropy loss
		input = torch.clone(input) 
		return {'input_ids': input, 'labels': labels}

	def __len__(self):
		return min(len(self.target_embeddings), len(self.query_embeddings))

filepath = '/home/bbadger/Desktop/retrieval_transformer_1024_200k.safetensors'
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
