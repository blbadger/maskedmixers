import torch
from einops import rearrange
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
from safetensors.torch import load_model
import warnings
from tqdm import tqdm
import time
warnings.simplefilter(action='ignore', category=FutureWarning) # for FSDP shard saving
warnings.simplefilter(action='ignore', category=UserWarning)


class LinearBlock(nn.Module):

    def __init__(self, dim, length, clm_mask=True):
        super().__init__()
        self.dim = dim
        self.length = length
        self.conv = nn.Conv1d(length, length, 1)
        self.clm_mask = clm_mask

    def forward(self, x: torch.tensor):
        if x.dim() > 3:
            x = rearrange(x, 'b p t f -> (b p) t f')

        # for CLM training, apply lower triangular mask to convolution weights
        if self.clm_mask:
            rearranged_shape = rearrange(self.conv.weight, 'f d p -> f (d p)').shape
            mask = torch.tril(torch.ones(rearranged_shape)).to(device)
            applied_mask = rearrange(self.conv.weight, 'f d p -> f (d p)') * mask
            self.conv.weight.data = rearrange(applied_mask, 'f (d p) -> f d p', p=1)

        residual = x
        x = self.conv(x) + residual
        return x

class LinearMixer(nn.Module):

    def __init__(self, n_vocab, dim, depth):
        super().__init__()
        self.wte = nn.Embedding(n_vocab, dim)
        self.mixerblocks = nn.ModuleList(
            [LinearBlock(
                dim = dim,
                length = tokenized_length,
                clm_mask=True,
                )
            for i in range(depth)]
            ).to(device)
        self.lm_head = nn.Linear(dim, n_vocab, bias=False)
        self.cel = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, input_ids, labels=None):
        x = input_ids
        x = x.to(device)
        x = self.wte(x)
        for block in self.mixerblocks:
            x = block(x)
        
        if labels is not None:
            x_prelim = x
            output = self.lm_head(x)
            labels = rearrange(labels, 'b p t -> b (p t)')
            output = rearrange(output, 'b t e -> b e t')
            shift_logits = output[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            #loss = self.cel(shift_logits, shift_labels)
            one_hots = torch.nn.functional.one_hot(shift_labels, num_classes=len(tokenizer)).transpose(1,2)
            converted_labels = torch.tensor(one_hots, requires_grad=False, dtype=torch.float)
            loss = self.mse(shift_logits, converted_labels)
            return loss, output, x_prelim
        else:
            return x

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/bbadger/Desktop/tiny_token_16k/tokenizer.json")
#tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_tinystories_16k")
#tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 2
print (tokenizer.eos_token)
n_vocab = len(tokenizer)# fails to properly read tokeinizer size
print (f"N vocab {n_vocab}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenized_length = 128

if __name__ == '__main__':
    dim = 16000
    model = LinearMixer(n_vocab, dim, 1).float()
    print (model)

    # cached dataset
    train_text = load_dataset("roneneldan/TinyStories", split="train")
    valid_text = load_dataset("roneneldan/TinyStories", split="validation")

    def debatch_input(input_data):
        output = []
        for i in range(len(input_data)):
            if input_data[i].dim() > 1:
                input_data[i] = input_data[i].unsqueeze(1)
                output += list(input_data[i])
        return output


    def batch_tokenize_input(train_text, test_text, length=20000, batch_size=1024):
        train_data, test_data = [], []
        max_length = 128

        for i in tqdm(range(0, length, batch_size)):
            input_ids = tokenizer.batch_encode_plus(
                train_text[i:i+batch_size]['text'],
                add_special_tokens=False,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length'
            ).input_ids
            train_data.append(input_ids)

        for i in range(0, len(test_text), batch_size):
            input_ids = tokenizer.batch_encode_plus(
                test_text[i:i+batch_size]['text'],
                add_special_tokens=False,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length'
            ).input_ids
            test_data.append(input_ids)

        train_data = debatch_input(train_data)
        test_data = debatch_input(test_data)

        return train_data, test_data

    train_data, test_data = batch_tokenize_input(train_text, valid_text)
    train_data, test_data = debatch_input(train_data), debatch_input(test_data)

    model.train()
    model = model.to('cuda')
    #trainer.train() 
    print (model.mixerblocks[0])
    def train_solver(model, train_data):
        train_batch = torch.stack(train_data[0:128], dim=0)
        print (train_batch.shape)
        train_batch = train_batch.to('cuda') 
        loss, output = model(train_batch, labels=train_batch) 
        print (f"Starting loss: {loss.item()}")
        loss.backward()
        start_time = time.time()    
        minimal_wte = torch.pinverse(model.wte.weight.grad) @ torch.zeros(model.wte.weight.shape).to('cuda')
        print (f'minimal wte found in {time.time() - start_time} seconds')
        start_time = time.time()
        minimal_conv = torch.pinverse(model.mixerblocks[0].conv.weight.grad) @ torch.zeros(model.mixerblocks[0].conv.weight.shape).to('cuda')
        print (f'minimal conv found ion {time.time() - start_time} seconds')
        start_time = time.time()
        minimal_lm = torch.pinverse(model.lm_head.weight.grad) @ torch.zeros(model.lm_head.weight.shape).to('cuda')
        print (f'minimal lm found in {time.time() - start_time} seconds')
        return minimal_params

    def normal_solve(model, train_data):
        input_batch = torch.stack(train_data[0:128], dim=0).to('cuda')
        loss, output, X = model(train_batch, labels=train_batch) 
        print (f"Starting loss: {(loss)}")
        beta_hat = torch.inverse(X.T @ X) @ X.T * target
        with torch.no_grad():
                model.lm_head.weight = torch.nn.Parameter(beta_hat)
                loss, output = model(train_batch, labels=train_batch) 
                print (f"Ending loss: {loss.item()}")
        return model

    def newton_iteration(model, train_data):
        train_batch = torch.stack(train_data[0:32], dim=0)
        train_batch = train_batch.to('cuda') 
        loss, output = model(train_batch, labels=train_batch) 
        print (f"Starting loss: {(loss)}")
        loss.backward() # gradients propegated to params
        slope = loss / model.lm_head.weight.grad
        print (slope)
        lm_head_min = model.lm_head.weight - slope
        with torch.no_grad(): model.lm_head.weight = torch.nn.Parameter(lm_head_min)
        loss, output = model(train_batch, labels=train_batch) 
        print (f"Ending loss: {loss}")
        return 

    newton_iteration(model, train_data)
    newton_iteration(model, train_data)





