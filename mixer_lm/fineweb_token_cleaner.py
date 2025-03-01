import os
from datasets import load_dataset, load_from_disk, Dataset
import argparse
import shutil

parser = argparse.ArgumentParser(description='Arg parser')
parser.add_argument('--path', type=str)

def clean(example):
    for col in list(example.keys()):
    	if col not in ['input_ids', 'attention_mask']:
    		del example[col]
    return example

if __name__ == '__main__':
	args = parser.parse_args()
	path = args.path
	dataset = load_from_disk(path)
	print ('pre clean', dataset[0])
	dataset = dataset.map(clean, batched=True)
	print ('post clean', dataset[0])
	dataset.save_to_disk(path + 'temp')
	shutil.rmtree(path)
	os.rename(path + 'temp', path)

