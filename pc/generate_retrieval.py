from llama_cpp import Llama
import json
from datasets import load_dataset
from tqdm import tqdm

model = Llama(
	model_path = '/path/to/llama-3-8b-instruct-Q8_0.gguf',
	n_gpu_layers = -1,
	chat_format='llama-3',
	verbose=False,
	n_ctx=4096
	)

train_text = load_dataset("roneneldan/TinyStories", split="train")
valid_text = load_dataset("roneneldan/TinyStories", split="validation")

batch_size = 16
outputs = []
for j in tqdm(range(200000)):
	output = model.create_chat_completion(
	      messages = [
	{"role": "system", "content": "You are an assistant for creating summaries for short stories."},
	          {
	              "role": "user",
	              "content": f"Give a brief one-sentence summary of the following story with no other text: {train_text[j]['text']}"
	          }
	]
	)
	outputs.append(output)

with open('path/to/dump/file.json', 'w') as f:
    json.dump(outputs, f)
