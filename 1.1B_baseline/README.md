# README

This is the baseline for our Language Model Catastrophic Forgetting, and it is also the experiment trying to reproduce the phenomenon in a language model scenario. 

The approaches to run this baseline:

## Approach 1 (Better)

1. Upload the `baseline.ipynb` to colab.
2. Use colab A100 with high RAM. You can also use other GPU type with lower RAM with turning batch size down.
3. Run the jupyter notebook.

## Approach 2 (Better)

This approach is suitable for GPU server who can have access to the Internet.

1. Run `baseline.py`

## Approach 3

This approach is suitable for GPU server who can't have access to the Internet. (e.g. Slurm)

1. Use `download_assets.py` to download the dataset and model
2. Modify the directories in `baseline.py` to the directories of your downloaded assets
3. Run `baseline.py`

