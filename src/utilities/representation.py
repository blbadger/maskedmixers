
from transformers import AutoTokenizer
from transformers import LlamaConfig, LlamaForCausalLM
import torch
import random
from torch import nn
from datasets import load_dataset
from safetensors.torch import load_model

device = 'cuda' if torch.cuda.is_available else 'cpu'
def octave(model, single_input, target_output, iterations, learning_rates):
    start_lr, end_lr = learning_rates
    original_input = single_input.clone()
    losses, i_arr = [], []

    for i in range(iterations):
        input_grad, loss = layer_gradient(model, single_input, target_output)
        single_input = single_input.detach()
        single_input -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)*input_grad
    return single_input

def layer_gradient(model, input_tensor, target, cosine_metric=False):
    input_tensor.requires_grad = True
    output = model(input_tensor)

    if cosine_metric:
        output, target = output[:, :, :].flatten(), target[:, :, :].flatten()
        loss = 1 - torch.abs(torch.dot(output, target)) / (torch.norm(output, p=2) * torch.norm(target, p=2))
  
    else:
        loss = torch.sum(torch.abs(target[:, :, :] - output[:, :, :]))
        
    # print (loss.item())
    loss.backward()
    gradient = input_tensor.grad
    return gradient, loss.item()

def generate_singleinput(model, embedding, target, lr=0.02): # 0.01
    random_input = torch.randn(embedding.shape).to(device)
    single_input = octave(model, random_input, target, 500, [lr, lr/10])
    return single_input

def hamming_metric(input_tokens, generated_tokens, tokenizer):
    # expects tokens to be pre-flattened
    assert len(input_tokens) == len(generated_tokens)
    count, card = 0, 0
    pad_token = tokenizer.encode(tokenizer.pad_token)[-1] # will be [2] for tiny_token_4k
    for i in range(len(input_tokens)):
        if input_tokens[i] == pad_token:
            continue
        else:
            card += 1
            if input_tokens[i] in generated_tokens[i]:
                count += 1
    return (card - count) / card