# README

This is the mixture experiment for our Language Model Catastrophic Forgetting.

The approaches to run this baseline:

## Approach 1 (Better)

1. Upload the `mixture.ipynb` to colab.
2. Use colab A100 with high RAM. You can also use other GPU type with lower RAM with turning batch size down.
3. Run the jupyter notebook.

## Approach 2 (Better)

This approach is suitable for GPU server who can have access to the Internet.

1. Run `mixture.py`

## Approach 3

This approach is suitable for GPU server who can't have access to the Internet. (e.g. Slurm)

1. Use `download_assets.py` to download the dataset and model
2. Modify the directories in `mixture.py` to the directories of your downloaded assets
3. Run `mixture.py`

