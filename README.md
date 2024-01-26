# The RAX Hypothesis

## Our Experiments

We are investigating the [RAX hypothesis](https://www.alignmentforum.org/posts/hjJXCn9GsskysDceS/what-s-up-with-llms-representing-xors-of-arbitrary-features) developed by Samuel Marks. It posits that LLMs linearly represent XORs of arbitrary features, even when there's no apparent reason to do so.

Currently, the following changes have been implemented in the original repository:
- Computing activations per batch instead of per statement
- Improving the way datasets are generated
- Adding new datasets
- Enhancing the methods for generating, storing, and loading activations
- Adding support for any HuggingFace transformer model
- Incorporating support for model revision to study the evolution of the RAX hypothesis during training using the Pythia models

Feel free to to get in touch in the #eliciting-latent-knowledge channel on the EleutherAI Discord server.

# Original README (not up to date)
This repository is associated to the paper <a href="https://arxiv.org/abs/2310.06824">*The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets*</a> by Samuel Marks and Max Tegmark. See also our <a href="https://saprmarks.github.io/geometry-of-truth/dataexplorer">interactive dataexplorer</a>.

(<a href="https://github.com/saprmarks/geometry-of-truth">View this page on github</a>.)

## Set-up

Navigate to the location that you want to clone this repo to, clone and enter the repo, and install requirements.
```
git clone git@github.com:butanium/geometry-of-truth.git
cd geometry-of-truth
pip install -r requirements.txt
```
Before doing anything, you'll need to generate activations for the datasets. You should have your own LLaMA weights stored on the machine where you cloned this repo. Put the absolute path for the directory containing your LLaMA weights in the file `config.ini` along with the names for the subdirectories containing weights for different scales. For example, my `config.ini` file looks like this:
```
[LLaMA]
weights_directory = /home/ubuntu/llama_hf/
7B_subdir = 7B
13B_subdir = 13B
30B_subdir = 30B
```
Once that's done, you can generate the LLaMA activations for the datasets you'd like to work with with a command like
```
python generate_acts.py --model 13B --layers 8 10 12 --datasets cities neg_cities --device cuda:0
```
These activations will be stored in the acts directory. If you want to save activations for all layers, simply use `--layers -1`.

## Files
This directory contains the following files:
* `dataexplorer.ipynb`: for generating visualizations of the datasets. Code for reproducing figures in the text is included.
* `few_shot.py`: for implementing the calibrated 5-shot baseline.
* `generalization.ipynb`: for training probes on one dataset and checking generalization to another. Includes code for reproducing the generalization matrix in the text.
* `interventions.ipynb`: for reproducing the causal intervention experiments from the text.
* `probes.py`: contains definitions of probe classes.
* `utils.py` and `visualization_utils.py`: utilities for managing datasets and producing visualizations. 


