# maskedmixers

![mixer](cover.png)

Code for the paper 'Masked Mixers for Language Generation and Retrieval', which you can read [here](https://arxiv.org/abs/2409.01482). Datasets and trained models will be added soon.

For a less formal version of this work written as a technical blog post, see [this page](https://blbadger.github.io/smaller-lms.html)

### TL;DR:
**Motivation:** Poor input representation accuracy in transformers, but much better accuracy in MLP-mixers adapted for causal language modeling (aka masked mixers)

**Finding:** Masked mixers are approximately as efficient learners of language generation relative to transformers but are far superior for retrieval.

### General Use

**Note that the `src` directory is currenty under construction: use `mixer_lm` for now**

Unless you want to replicate a specific experiment, use the `src` directory to train, run, and evaluate mixers and other related models.

### For Experimental Replication

Use the `mixer_lm` directory for replication of experiments.
