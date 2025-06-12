import torch
from einops import rearrange
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
from safetensors.torch import load_model
import warnings
from tqdm import tqdm

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

    def forward(self, input_ids, labels=None):
        x = input_ids
        x = x.to(device)
        x = self.wte(x)
        for block in self.mixerblocks:
            x = block(x)
        
        if labels is not None:
            output = self.lm_head(x)
            labels = rearrange(labels, 'b p t -> b (p t)')
            output = rearrange(output, 'b t e -> b e t')
            shift_logits = output[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.cel(shift_logits, shift_labels)
            return loss, output
        else:
            return x

def train_solver(model):
    gradient = model.grad()
    # equation: 0 = gradient
    return minimal_params


def newton_method(model, train_data, gradient):
    pass

def obtain_gradients(model):
    gpu_count = torch.cuda.device_count()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = model.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    optimizer = Adam(ddp_model.parameters(), lr=1e-4)
    ddp_model.train()

    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        total_loss = 0
        total_mse_loss = 0
        accumulation_loss = torch.tensor(0.).to(device_id)
        for step, batch in enumerate(dataloader):
            if len(batch) < batch_size:
                break 
            
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
                batch = batch.to(device_id) # discard class labels
                output = ddp_model(batch) 
                loss = loss_fn(output, batch)

                loss_size = loss.shape
                loss = torch.sum(loss)
                total_loss += loss.item()
            scaler.scale(loss).backward()
            if (step + 1) % gradient_accumulations == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if rank == 0:
            checkpoint_path = f'/home/bbadger/Desktop/churches_unetwidedeep/epoch_{epoch}'
            if epoch % 100 == 0: torch.save(ddp_model.state_dict(), checkpoint_path)
            tqdm.write(f"Epoch {epoch} completed in {time.time() - start_time} seconds")
            tqdm.write(f"Average Loss: {round(total_loss / step, 5)}")
            tqdm.write(f"Loss shape: {loss_size}")
        dist.barrier()

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


    def batch_tokenize_input(train_text, test_text, length=2000000, batch_size=1024):
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

    train_autoencoder(model, dataset='landscapes')