# Causal Representation Learning via multi-node interventions

This repository contains the code to reproduce all of our experimental details.

To do so, run:
```console
python experiment_main.py
```

The most important flags are: 
- ```--models``` to select which models to compare (choose from ```ours,icrl,ica```).
- ```--scm_id``` to select which underlying SCM model is used.
- ```--d``` to pass the number of nodes in the underlying model to generate the data.