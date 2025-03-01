from datasets import load_dataset
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import os
from transformers import AutoTokenizer
import time
import torch
from transformers import AutoTokenizer, BatchEncoding

dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT")
old_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

class TextDataset(torch.utils.data.Dataset):
    """
    Create a Dataset object from a file consisting of lines of strings
    """
    def __init__(self, file_path, batch_size, truncation_index=2000000):
        super().__init__()
        self.batch_size = batch_size
        self.lines = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                self.lines.append(line.strip())  # Remove newline characters
        self.lines = self.lines[:truncation_index]

        self.line_batches = []
        batches = len(self.lines) // batch_size
        for i in range(batches):
            self.line_batches.append(self.lines[i*batch_size: i*(batch_size+1)])
        print (f'Batches to tokenize: {len(self.line_batches)}')

    def __len__(self):
        return len(self.line_batches)

    def __getitem__(self, idx):
        print (f"Tokenizing batch {idx}") if idx % 100 == 0 else None
        batch = self.line_batches[idx]
        return batch

# Create the dataset, and process the full file. 
# dataset = TextDataset(dataset, batch_size=1024)
# DataLoader for efficient batch processing
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

def get_training_corpus(dataset, batch_size=1):
    #dataset = dataset["train"]
    for i in range(0, len(dataset)):
        sample = dataset[i]
        yield sample['text']

training_corpus = get_training_corpus(dataset)
# print (next(training_corpus))

# Train the new tokenizer
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 128000)
tokenizer.save_pretrained("/home/bbadger/Desktop/tokenizer_fineweb_128k")
print ("Tokenizer saved")
