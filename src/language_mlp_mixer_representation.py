import os

# run on only one GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from transformers import AutoTokenizer, AutoTokenizer
import torch
import random
from torch import nn
import random
from prettytable import PrettyTable
from datasets import load_dataset
from functools import partial 
from einops import rearrange
from safetensors.torch import load_model
from utilities.mixer_models import LanguageMixer
from utilities.representation import hamming_metric, generate_singleinput

device = 'cuda' if torch.cuda.is_available() else 'cpu'
manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)

class AbbreviatedMixer(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        for i in range(8):
            x = self.model.mixerblocks[i](x)
        return x


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tiny_token_4k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    short_inputs=False
    dims = [512]
    for dim in dims:
        root = '/path/to/root/' # directory with model checkpoints to test
        hammings = []
        # optional: sort checkpoints start -> end
        sorted_dirs = sorted(os.listdir(root))[:-1] # remove 'llama.py'
        sorted_dirs.sort(key=lambda dir: int(dir[11:]))
        for i in range(0, len(sorted_dirs), 1):
            dir = sorted_dirs[i]
            tokenized_length = 512
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = LanguageMixer(n_vocab, dim, 8).float().to(device)

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

            tokenizer.pad_token = tokenizer.eos_token
            hamming_metrics = []
            
            # assumes file saved using safetensors model loading
            load_model(model, root + dir + '/model.safetensors')
            for prompt in prompts:
                tokens = tokenizer.encode(
                      prompt,
                      add_special_tokens=False,
                      return_tensors='pt',
                      truncation=True,
                      padding='max_length', 
                      max_length=tokenized_length,
                      ).to(device)

                og_model = model
                a_model = AbbreviatedMixer(model)
                embedding = og_model.wte(tokens)

                shifted_embedding = embedding + 0.05*torch.randn(embedding.shape).to(device)
                embedding_weight = og_model.wte.weight.float() # convert to float in case model is in 16-bit precision
                inverse_embedding = torch.linalg.pinv(embedding_weight.cpu()).to(device)
                logits = torch.matmul(shifted_embedding.float(), inverse_embedding.float()) # invert embedding transformations
                tokens = torch.argmax(logits, dim=2)[0]

                a_model.eval()
                with torch.no_grad():
                    shifted_target_tensor = a_model(shifted_embedding).to(device)
                    target_tensor = a_model(embedding).to(device)

                embedding = embedding.detach()
                generated_input = generate_singleinput(a_model, embedding, target_tensor)
                g_input = generated_input
                generated_target_tensor = a_model(g_input).to(device)
                logits = torch.matmul(generated_input, inverse_embedding)
                topk_k = 5
                generated_tokens = torch.topk(logits, topk_k)[1][0] # indicies of topk of tensor [length, topk_tokens]

                for i in range(1):
                    output = tokenizer.decode([o[i] for o in generated_tokens])
                    # print (output)
                    break

                metric = hamming_metric(tokens, generated_tokens, tokenizer)
                print (metric)
                hamming_metrics.append(metric)
            hammings.append(hamming_metrics)
            print (f'Hamming metrics for dim {d}: ', hamming_metrics)
        print (hammings)