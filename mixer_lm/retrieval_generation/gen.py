from llama_cpp import Llama
import json
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='Driver args')
parser.add_argument('--start', type=int)
parser.add_argument('--stop', type=int)
parser.add_argument('--output_path', type=str)

if __name__ == '__main__':
	args = parser.parse_args()
	model = Llama(
		model_path = '/home/bbadger/Desktop/llama-3-8b-instruct-Q8_0.gguf',
		n_gpu_layers = -1,
		chat_format='llama-3',
		verbose=False,
		n_ctx=4096
		)

	# train_path = "/home/bbadger/Desktop/fineweb-edu-tokenized-train-c512"
	train_path = "/home/bbadger/Desktop/finemath-4-tokenized-train-c512-8k"
	train_text = load_from_disk(train_path)
	tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
	tokenizer.pad_token = tokenizer.eos_token

	outputs = []
	for j in tqdm(range(args.start, args.stop)):
       		text = train_text[j]['text']
        	text = tokenizer.decode(train_text[j]['input_ids']).strip('<|end_of_text|>') # strip padding
        	output = model.create_chat_completion(
              		messages = [ 
        		{"role": "system", "content": "You are an assistant for creating summaries for short stories."},
                  		{
                      	"role": "user",
                      	"content": f"Give a brief few-word summary of the following with no other output, do not begin with 'Here is a summary...'. Text: {text}"
                  	}
        		]
        	)
	       # print (text,'\n Summary: \n', output['choices'][0]['message']['content'])
       		outputs.append(output)

	output_path = args.output_path + f'_{args.start}_{args.stop}.json'
	with open(output_path, 'w') as f:
	    json.dump(outputs, f)
