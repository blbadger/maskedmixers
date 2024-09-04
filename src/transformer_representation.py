import os

# run on only one GPU if more than one are present
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoTokenizer
from transformers import LlamaConfig, LlamaForCausalLM
import torch
import random
from torch import nn
from datasets import load_dataset
from safetensors.torch import load_model
from utilities.representation import generate_singleinput, hamming_metric

device = 'cuda' if torch.cuda.is_available() else 'cpu'
manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)

class AbbreviatedModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # Matrix mult instead of embedding to prevent type incompatibility
        position_ids = torch.tensor([[i for i in range(x.shape[1])]]).to(device)

        for i in range(8):
            x = self.model.model.layers[i](x, position_ids=position_ids)[0]

        return x

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    dims = [256] # specify dimensions of models of interest
    short_inputs=False
    for dim in dims:
        root = '/path/to/root/' # directory with model checkpoints to test
        hammings = []
        # optional: sort checkpoints start -> end
        sorted_dirs = sorted(os.listdir(root))[:-1] # remove 'llama.py'
        sorted_dirs.sort(key=lambda dir: int(dir[11:]))
        for dir in sorted_dirs:
            tokenized_length = 512
            
            llama_config_kwargs = {
                'hidden_size': dim,
                'intermediate_size': 4*dim,
                'num_hidden_layers': 8,
                'num_attention_heads': 32,
                'vocab_size': 4096
            }

            # Initializing a LLaMA model
            configuration = LlamaConfig(**llama_config_kwargs)

            # Initializing a model from the llama-7b style configuration
            model = LlamaForCausalLM(configuration).float()

            train_text = load_dataset("roneneldan/TinyStories", split="train")
            valid_text = load_dataset("roneneldan/TinyStories", split="validation")

            if short_inputs:
                prompts = [
                'Mario, the Idea, versus Mario, the Man', 
                'An apple a day keeps the doctor away',
                'Four score and seven years ago, our forefathers',
                'It was the best of times, it was the worst of times.',
                'Call me Ishmael. Some years ago-never mind how long',
                'Seven stars and seven stones and one white tree',
                'Attention mechanisms that confer selective focus',
                'Numbers are the things themselves.',
                'Weather, whither by the whithywindle',
                'Mr and Mrs Dursley of number four, Privet Drive, were proud to say that they were perfectly normal'
                ]
            else:
                prompts = [text for text in valid_text[:10]['text']]

            og_model = model

            # load model weights
            load_model(model, root + dir + '/model.safetensors')

            tokenizer.pad_token = tokenizer.eos_token
            hamming_metrics = []
            for prompt in prompts:
                tokens = tokenizer.encode(
                      prompt,
                      add_special_tokens=False,
                      return_tensors='pt',
                      truncation=True,
                      padding='max_length', 
                      max_length=tokenized_length,
                      ).to(device)

                
                a_model = AbbreviatedModel(model).to(device)
                embedding = og_model.model.embed_tokens(tokens)
                shifted_embedding = embedding + 0.05*torch.randn(embedding.shape).to(device)
                embedding_weight = og_model.model.embed_tokens.weight.float() # convert to float in case model is in 16-bit precision
                inverse_embedding = torch.linalg.pinv(embedding_weight.cpu()).to(device)
                logits = torch.matmul(shifted_embedding.float(), inverse_embedding.float()) # invert embedding transformation
                tokens = torch.argmax(logits, dim=2)[0]
                output = tokenizer.decode(tokens)

                a_model.eval()
                with torch.no_grad():
                    shifted_target_tensor = a_model(shifted_embedding.to(device)).to(device)
                    target_tensor = a_model(embedding).to(device)

                random_embedding = embedding.detach()
                generated_input = generate_singleinput(a_model, random_embedding, target_tensor)
                g_input = generated_input

                generated_target_tensor = a_model(g_input).to(device)
                logits = torch.matmul(generated_input, inverse_embedding)
                topk_k = 5
                generated_tokens = torch.topk(logits, topk_k)[1][0] # indicies of topk of tensor [length, topk_tokens]
                metric = hamming_metric(tokens, generated_tokens, tokenizer)
                hamming_metrics.append(metric)
                print (metric)
            hammings.append(hamming_metrics)

            print (f'Hamming metrics for dim {d}: ', hamming_metrics)
        print (hammings) # or save for further processing

