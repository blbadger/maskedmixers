from llama_cpp import Llama
import json
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, Dataset

model = Llama(
	model_path = '/home/bbadger/Desktop/llama-3-8b-instruct-Q8_0.gguf',
	n_gpu_layers = -1,
	chat_format='llama-3',
	verbose=False,
	n_ctx=4096
	)

# train/validation set: first 100k train examples, first 10k validation examples
#train_text = load_dataset("roneneldan/TinyStories", split="train")
#valid_text = load_dataset("roneneldan/TinyStories", split="validation")
train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
train_text = load_from_disk(train_path)
tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")

batch_size = 16
outputs = []
for j in tqdm(range(10)):
	#text = train_text[j]['text']
	text = tokenizer.decode(train_text[j]['input_ids']).strip('<|end_of_text|>')
	output = model.create_chat_completion(
	      messages = [
	{"role": "system", "content": "You are an assistant for creating summaries for short stories."},
	          {
	              "role": "user",
	              "content": f"Give a brief one-sentence summary of the following with no other output, do not begin with 'Here is a summary...'. Text: {text}"
	          }
	]
	)
	print (text,'\n Summary: \n', output['choices'][0]['message']['content'])
	outputs.append(output)

with open('/home/bbadger/Desktop/train_output_300_350k.json', 'w') as f:
    json.dump(outputs, f)
