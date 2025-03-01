import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import random
import numpy as np
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import math, random, time
import prettytable
from prettytable import PrettyTable
from datasets import load_dataset

import einops
from functools import partial 
from einops import rearrange, reduce
from safetensors.torch import load_model, save_model, load_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (device)

manualSeed = 1
random.seed(manualSeed)
torch.manual_seed(manualSeed)

def octave(single_input, target_output, iterations, learning_rates):
    start_lr, end_lr = learning_rates
    original_input = single_input.clone()
    losses, i_arr = [], []

    for i in range(iterations):
        # input_grad, loss = layer_gradient(model, single_input, target_output)
        input_grad, loss = layer_gradient(model, single_input, target_output)
        single_input = single_input.detach()
        single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad
    return single_input

def generate_singleinput(model, target, lr=2): # 0.02
    random_input = torch.randn(embedding.shape).to(device)
    single_input = octave(random_input, target, 500, [lr, lr/10])
    return single_input

def layer_gradient(model, input_tensor, target, cosine_metric=False):
    input_tensor.requires_grad = True
    output = a_model(input_tensor)

    if cosine_metric:
        last = 2201
        output, target = output[:, :, :].flatten(), target[:, :, :].flatten()
        loss = 1 - torch.abs(torch.dot(output, target)) / (torch.norm(output, p=2) * torch.norm(target, p=2))
  
    else:
        loss = torch.sum(torch.abs(target[:, :, :] - output[:, :, :]))
        
    # print (loss.item())
    loss.backward()
    gradient = input_tensor.grad
    return gradient, loss.item()

def feature_gradient(model, input_tensor, index=0):
    input_tensor.requires_grad = True # usually only necessary once
    output = a_model(input_tensor)
    # assumes dims of [batch, token, hidden_dim]
    loss = torch.sum(100 - output[:, :, index])
    loss.backward()
    gradient = input_tensor.grad
    return gradient, loss.item()


class AbbreviatedModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # Matrix mult instead of embedding to prevent type incompatibility
        position_ids = torch.tensor([[i for i in range(x.shape[1])]]).to(device)

        for i in range(len(self.model.model.layers)):
            x = self.model.model.layers[i](x, position_ids=position_ids)[0]

        return x


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def hamming_metric(input_tokens, generated_tokens):
    # expects tokens to be pre-flattened
    assert len(input_tokens) == len(generated_tokens)
    count, card = 0, 0
    pad_token = tokenizer.encode(tokenizer.pad_token)[-1] # will be [2]
    for i in range(len(tokens)):
        if input_tokens[i] == pad_token:
            continue
        else:
            card += 1
            if input_tokens[i] in generated_tokens[i]:
                count += 1
    return (card - count) / card


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_8k")
    tokenizer.pad_token = tokenizer.eos_token
    n_vocab = len(tokenizer)
    dims = [512]
    for d in dims:
        root = '/home/bbadger/Desktop/tinystories/llama_256_longs/'
        hammings = []
        sorted_dirs = sorted(os.listdir(root))[:-1] # remove 'llama.py'
        sorted_dirs.sort(key=lambda dir: int(dir[11:]))
        for dir in sorted_dirs:
            tokenized_length = 1024
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dim = d
            llama_config_kwargs = {
                'hidden_size': dim,
                'intermediate_size': 4*dim,
                'num_hidden_layers': 16,
                'num_attention_heads': 4,
                'vocab_size': 8000
            }

            # Initializing a LLaMA model
            configuration = LlamaConfig(**llama_config_kwargs)

            # Initializing a model from the llama-7b style configuration
            model = LlamaForCausalLM(configuration).float()
            print (model)

            train_text = load_dataset("roneneldan/TinyStories", split="train")
            valid_text = load_dataset("roneneldan/TinyStories", split="validation")

            # prompt = train_text[0]['text']
            # print (prompt)
            # prompts = [
            # 'Mario, the Idea, versus Mario, the Man', 
            # 'An apple a day keeps the doctor away',
            # 'Four score and seven years ago, our forefathers',
            # 'It was the best of times, it was the worst of times.',
            # 'Call me Ishmael. Some years ago-never mind how long',
            # 'Seven stars and seven stones and one white tree',
            # 'Attention mechanisms that confer selective focus',
            # 'Numbers are the things themselves.',
            # 'Weather, whither by the whithywindle',
            # 'Mr and Mrs Dursley of number four, Privet Drive, were proud to say that they were perfectly normal'
            # ]

            # prompts = [text for text in valid_text[:50]['text']]
            valid_text = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", split="train", streaming=True)
            prompts = []
            count = 0
            for example in valid_text:
                count += 1
                if count > 50:
                    break
                prompts.append(example['text'])
            og_model = model

            # for safetensors
            print (root + dir)
            # load_model(model, root + dir + '/model.safetensors')
            load_model(model, '/home/bbadger/Desktop/model.safetensors')
            print ('model_loaded')

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
                      padding_side='right'
                      ).to(device)

                
                a_model = AbbreviatedModel(model).to(device)
                embedding = og_model.model.embed_tokens(tokens)
                shifted_embedding = embedding + 0.05*torch.randn(embedding.shape).to(device)
                print (f'Shifted embedding distance: {torch.sum(torch.abs(embedding - shifted_embedding))}')
                embedding_weight = og_model.model.embed_tokens.weight.float() # convert to float in case model is in 16-bit precision
                inverse_embedding = torch.linalg.pinv(embedding_weight.cpu()).to(device)
                print ('inverse embedding computed')
                logits = torch.matmul(shifted_embedding.float(), inverse_embedding.float()) # invert embedding transformations
                tokens = torch.argmax(logits, dim=2)[0]
                output = tokenizer.decode(tokens)

                a_model.eval()
                with torch.no_grad():
                    shifted_target_tensor = a_model(shifted_embedding.to(device)).to(device)
                    target_tensor = a_model(embedding).to(device)
                print (f'Shifted output distance: {torch.sum(torch.abs(shifted_target_tensor - target_tensor))}')

                embedding = embedding.detach()
                generated_input = generate_singleinput(a_model, target_tensor)
                g_input = generated_input
                generated_target_tensor = a_model(g_input).to(device)
                print (f'Generated output distance: {torch.sum(torch.abs(generated_target_tensor - target_tensor))}')                                                  
                logits = torch.matmul(generated_input, inverse_embedding)
                topk_k = 5
                generated_tokens = torch.topk(logits, topk_k)[1][0] # indicies of topk of tensor [length, topk_tokens]\

                for i in range(1):
                    output = tokenizer.decode([o[i] for o in generated_tokens])
                    print (output)
                    break

                # print ('\n')
                # print (generated_tokens.shape)
                metric = hamming_metric(tokens, generated_tokens)
                hamming_metrics.append(metric)
                print (metric)
            hammings.append(hamming_metrics)

            print (f'Hamming metrics for dim {d}: ', hamming_metrics)
            break
        print (hammings)

