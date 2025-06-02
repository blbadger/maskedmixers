# maskedmixers

![mixer](cover.png)

Code for the paper 'Masked Mixers for Language Generation and Retrieval', which you can read [here](https://arxiv.org/abs/2409.01482). Datasets and trained models will be added soon.

For a less formal version of this work written as a technical blog post, see [this page](https://blbadger.github.io/smaller-lms.html)

### Paper TL;DR:
**Motivation:** Poor input representation accuracy in transformers, but much better accuracy in MLP-mixers adapted for causal language modeling (aka masked mixers)
**Finding:** Masked mixers are approximately as efficient learners of language generation relative to transformers but are far superior for retrieval.

### General Use

To use this code, spin up a virtual environment and install the necessary requirements via `pip install -r requirements.txt`. If you expect to build additional features using dependency-heavy libraries like `vllm` or else want to limit the number of virtual environments you use among many repos, using `uv` to manage python packages and installing more recent compatible libraries via `requirements_updated.txt` is recommended (ie `uv pip install -r requirements.txt`). 

Depending on your CUDA driver and runtime versions, you may receive the following error resulting from an undefined symbol in your ncclCommRegister upon attempting to install from `requirements.txt`:

```
Traceback (most recent call last):
  File "/home/bbadger/experiments/maskedmixers/mixer-venv/bin/torchrun", line 5, in <module>
    from torch.distributed.run import main
  File "/home/bbadger/experiments/maskedmixers/mixer-venv/lib/python3.10/site-packages/torch/__init__.py", line 290, in <module>
    from torch._C import *  # noqa: F403
ImportError: /home/bbadger/experiments/maskedmixers/mixer-venv/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so: undefined symbol: ncclCommRegister
```

If you do, simply install `torch==2.1.1` and then re-install (see further [this torch issue](https://github.com/pytorch/pytorch/issues/119932) for more information).

This repo expects CUDA runtime API 12.x and Python 3.10.12, although it is compatible with other minor versions of 3.10 and should be compatible with a few other major versions as well (this has not been tested). 

If you are using a different Python version, it is recommended that you use a python environment manager (`pyenv` etc.) to install 3.10.12 before spinning up the venv and installing dependencies. If you currently have a different CUDA major version such as 11.x, it is recommended that you use build a docker container with CUDA 12.2 or 12.3 rather than attempt to upgrade your system CUDA version, as doing so has a tendency to break existing libraries that use CUDA in very obscure ways.


### For Experimental Replication

Use the `mixer_lm` directory for replication of experiments.

**Note that the `src` directory is currenty under construction: use `mixer_lm` for now**

A more user-friendly version is `src` directory to train, run, and evaluate mixers and other related models.

