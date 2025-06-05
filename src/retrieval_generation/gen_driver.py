import argparse
import subprocess
import torch

template = "CUDA_VISIBLE_DEVICES={} python gen.py --start {} --stop {} --output_path {}&\n"

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--n_gpus', type=int)
parser.add_argument('--output_path', type=str)
parser.add_argument('--n_samples', type=int)
parser.add_argument('--start_index', type=int)
print ('parser initialized')

if __name__ == "__main__":
	args = parser.parse_args()
	n_gpus = args.n_gpus
	n_samples = args.n_samples
	output_path = args.output_path
	bash_string = ""
	assert n_samples % n_gpus == 0, 'Number of samples must be divisible by number of GPUs used'

	for gpu_index in range(n_gpus):
		selected = int(n_samples// n_gpus)
		start = gpu_index*selected + args.start_index
		stop = gpu_index*selected + selected + args.start_index
		bash_string += template.format(gpu_index, start, stop, output_path)

	bash_string = bash_string[:-2] # strip '&/n' from last templated entry
	print (f'Running string: {bash_string}')
	subprocess.run(bash_string, shell=True)
