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

    def __init__(self, n_vocab, dim, depth, mse_loss=False):
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
        self.mse_loss = mse_loss
        self.cel = nn.CrossEntropyLoss(reduction='none')
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
            if not self.mse_loss:
                loss = self.cel(shift_logits, shift_labels)
            else:
                # convert labels to one hots, compute mse
                one_hots = torch.nn.functional.one_hot(shift_labels, num_classes=len(tokenizer)).transpose(1,2) 
                converted_labels = torch.tensor(one_hots, requires_grad=False, dtype=torch.float)
                loss = self.mse(shift_logits, converted_labels)
            return loss, output, x_prelim
        else:
            return x

tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/bbadger/Desktop/tiny_token_4k/tokenizer.json")
#tokenizer = AutoTokenizer.from_pretrained("/home/bbadger/Desktop/tokenizer_tinystories_16k")
#tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = 2
print (tokenizer.eos_token)
n_vocab = len(tokenizer) # actually a very small n=66 tokenizer
print (f"N vocab {n_vocab}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenized_length = 128

if __name__ == '__main__':
    dim = 100
    model = LinearMixer(n_vocab, dim, 1, mse_loss=True).double()
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

    def batch_tokenize_input(train_text, test_text, length=2000, batch_size=1024):
        train_data, test_data = [], []
        max_length = tokenized_length

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

    def grad_descent(model, train_data, lr=0.005):
        for i in range(1000):
            train_batch = torch.stack(train_data[0:100], dim=0).to('cuda')
            loss, output, _ = model(train_batch, labels=train_batch)
            torch.mean(loss).backward() # gradients propegated to params
            with torch.no_grad(): 
                model.lm_head.weight -= lr * model.lm_head.weight.grad
                output = model(train_batch, labels=train_batch)
                print (f"Grad descent iteration {i} loss: {torch.mean(loss)} \n")
            model.zero_grad()
        return 
    
    @torch.no_grad()
    def normal_solve(model, train_data):
        train_batch = torch.stack(train_data[0:100], dim=0).to('cuda')
        loss, output, X = model(train_batch, labels=train_batch)
        print (f"Starting loss: {loss.item()}")
        target = torch.nn.functional.one_hot(train_batch, num_classes=len(tokenizer)).to(torch.double).squeeze(1)
        prefix = torch.pinverse(X)
        print (prefix.shape, target.shape)
        beta_hat = torch.mean(prefix @ target, dim=0).T
        print (f'Optimal params computed: {beta_hat.shape}')
        print (beta_hat.shape, model.lm_head.weight.shape)
        with torch.no_grad():
                model.lm_head.weight = torch.nn.Parameter(beta_hat)
                loss, output, X = model(train_batch, labels=train_batch) 
                print (f"Ending loss: {loss.item()}")
        return model

    def newton_iterations(model, train_batch, loss_constant=0.01):
        train_batch = torch.stack(train_data[0:1], dim=0).to('cuda')
        for i in range(5):
                model.zero_grad()
                loss, output, _ = model(train_batch, labels=train_batch)
                loss -= loss_constant # subtract suspected irreducible loss so root exists
                print (f"Starting loss: {(loss)}")
                loss.backward()
                print (model.lm_head.weight.grad, torch.norm(model.lm_head.weight.grad))
                loss_term = torch.pinverse(model.lm_head.weight.grad) * loss 
                print (torch.norm(loss_term))
                model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight - loss_term.T)
                with torch.no_grad(): 
                        loss, output, _ = model(train_batch, labels=train_batch)
                        print (f"Ending loss: {loss-loss_constant} \n")
        return 

    def newton_components(model, train_data, loss_constant=0.1):
        train_batch = torch.stack(train_data[0:100], dim=0).to('cuda')
        for i in range(10):
            print (f'Iteration {i}')
            loss, output, _ = model(train_batch, labels=train_batch)
            loss -= loss_constant # subtract suspected irreducible loss so root exists
            loss_terms = []
            for j in range(tokenized_length-1):
                for k in range(len(train_batch)):
                    loss[k][j].backward(retain_graph=True)
                    loss_term = torch.pinverse(model.lm_head.weight.grad) * loss[k][j]
                    loss_terms.append(loss_term)
                    model.zero_grad()
            for loss_term in loss_terms:
                model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight - loss_term.T)
            print (f"Loss: {(torch.mean(loss))}")
        return 

    def newton_components_recalculated(model, train_data, loss_constant=0.9):
        train_batch = torch.stack(train_data[0:100], dim=0).to('cuda')
        with torch.no_grad():
            loss, output, _ = model(train_batch, labels=train_batch)
            print (f"Starting loss: {torch.mean(loss)}")
        print (loss.rearranged_shape)
        for i in range(100):
            for j in range(tokenized_length-1):
                for k in range(len(train_batch)):
                    loss, output, _ = model(train_batch, labels=train_batch)
                    loss -= loss_constant # subtract suspected irreducible loss so root exists
                    loss[k][j].backward()
                    loss_term = torch.pinverse(model.lm_head.weight.grad) * loss[k][j] 
                    with torch.no_grad():
                        model.lm_head.weight-= loss_term.T
                    model.zero_grad()
            print (f"Step {i} Loss: {(torch.mean(loss))}")
        return

    normal_solve(model, train_data)
    # newton_iterations(model, train_data)
    # newton_components(model, train_data)
    # newton_components_recalculated(model, train_data)
    # grad_descent(model, train_data)





