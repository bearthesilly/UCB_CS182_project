This experiment is for running the 7B baseline experiment. Here we train on ONLY MATH dataset and then on ONLY HotPotQA dataset, and we measure the loss as we are training HotPotQA on both HotPotQA and MATH. We demonstrate that catastrophic forgetting has occured. 

Run the `baseline_file.py`. This will output all the files into the relevant folder defined as RESULTS_DIR: ("out"). Run it simply with `python baseline_file.py`.
