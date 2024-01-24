# Identifying Linearly-Mixed Causal Representations from Multi-Node Interventions
[![Paper](https://img.shields.io/static/v1.svg?logo=arxiv&label=Paper&message=Open%20Paper&color=green)](https://arxiv.org/abs/2311.02695)

This is the official accompanying code repository for the paper **Identifying Linearly-Mixed 
Causal Representations from Multi-Node Interventions** by Simon Bing, Urmi Ninad,
Jonas Wahl and Jakob Runge.

## Requirements
This code was developed with Python 3.10 and PyTorch 2.0.1. Install the required
dependencies by running
```console
conda env create -f environment.yml
```
and activate the environment by running
```console
conda activate multi_node_crl
```

## Experiments
Our experiments can all be reproduced from a single script.
To do so, run:
```console
python experiment_main.py
```

The most important flags are: 
- ```--models``` to select which models to compare (choose from ```ours,icrl,ica```).
- ```--scm_id``` to select which underlying SCM model is used.
- ```--d``` to pass the number of nodes in the underlying model to generate the data.
