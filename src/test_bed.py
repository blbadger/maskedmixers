import torch
from einops import rearrange
import torch.nn as nn
from utilities.mixer_models import LanguageMixer

tokenized_length = 3
depth = 2
dim = 10
n_vocab=4096
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LanguageMixer(n_vocab, dim, depth, tokenized_length=tokenized_length).float().to(device)

one = torch.tensor([[[1, 2, 3]]]).to(device)
two = torch.tensor([[[1, 4, 3]]]).to(device)
print (model(one, labels=one))
print (model(two, labels=two))