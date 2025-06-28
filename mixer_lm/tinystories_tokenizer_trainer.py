from datasets import load_dataset
from transformers import AutoTokenizer
import torch

dataset = load_dataset("roneneldan/TinyStories", split="train")
print (len(dataset))
#dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", name="sample-10BT")
#old_tokenizer = AutoTokenizer.from_pretrained("/home/bbadger//tiny_token_8k")
old_tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")

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


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset["text"][idx]

# Create the dataset, and process the full file. 
#dataset = TinyDataset(dataset)
#DataLoader for efficient batch processing
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

def get_training_corpus(dataset):
    dataset = dataset["train"]
    for i in range(len(dataset)):
        sample = dataset[i]
        print (len(sample['markdown']))
        yield sample['markdown']

training_corpus = get_training_corpus(dataset)
# Train the new tokenizer
iterable_dataset = [i["text"] for i in dataset]
print (len(iterable_dataset))
tokenizer = old_tokenizer.train_new_from_iterator(iterable_dataset, 8192)
tokenizer.save_pretrained("/home/bbadger/Desktop/tiny_token_8k")
print ("Tokenizer saved")
