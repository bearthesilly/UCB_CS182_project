# Catastrophic Forgetting Experiments with Data Mixing Strategies

This directory contains experiments investigating catastrophic forgetting mitigation through different data mixing strategies during sequential task training.

## Overview

We train a TinyLlama-1.1B model on two tasks sequentially:
1. **Phase A (MATH)**: Train on mathematical problem-solving
2. **Phase B (HotpotQA)**: Train on question-answering with context

During Phase B training, we mix in a small percentage (10%) of Phase A data using different selection strategies to mitigate catastrophic forgetting.

## Quick Start

### Running All Experiments

All experiments can be run using the provided shell script:

```bash
bash run_experiments.sh
```

This will execute four different data mixing strategies:
- **Human-Ranked Difficult**: Select the hardest MATH problems (by difficulty level)
- **Random**: Randomly select MATH problems
- **Human-Ranked Easy**: Select the easiest MATH problems (by difficulty level)
- **Low-Loss**: Select MATH problems with lowest loss (model finds them easiest)
- **High-Loss**: Select MATH problems with highest loss (model finds them difficult)


### Generating Plots

After running the experiments, you'll need to manually plot the results. The plotting command is commented out in `run_experiments.sh` because it depends on the output file names.

To plot your results:

```bash
source .venv/bin/activate && python plot_forgetting_history.py -o output.png \
  drive/MyDrive/forgetting_history_MATH_to_HotpotQA_difficulty_hard_10pct_bf16.json \
  drive/MyDrive/forgetting_history_MATH_to_HotpotQA_random_10pct_bf16.json \
  drive/MyDrive/forgetting_history_MATH_to_HotpotQA_difficulty_easy_10pct_bf16.json \
  drive/MyDrive/forgetting_history_MATH_to_HotpotQA_loss_ascending_10pct_bf16.json
```

Or use wildcards to plot all results:

```bash
source .venv/bin/activate && python plot_forgetting_history.py -o output.png \
  drive/MyDrive/forgetting_history_*.json
```

## Main Scripts

### `difficulty_mixing.py`

The main experimental script that:
1. Loads the TinyLlama-1.1B model with LoRA adapters
2. Trains on MATH dataset (Phase A) or loads pre-trained adapter
3. Selects a subset of MATH data using the specified strategy
4. Mixes selected MATH data (10%) with HotpotQA data
5. Trains on the mixed dataset (Phase B)
6. Tracks validation loss on both tasks during Phase B training
7. Saves results to JSON and generates individual plots

**Key Parameters:**
- `--selection-method`: How to select Phase A data for mixing
  - `difficulty_hard`: Select hardest problems by human difficulty rating
  - `difficulty_easy`: Select easiest problems by human difficulty rating
  - `random`: Random selection
  - `loss_descending`: Select problems with highest loss (hardest for model)
  - `loss_ascending`: Select problems with lowest loss (easiest for model)

**Configuration Variables (in script):**
- `MIX_PERCENTAGE = 0.1`: Mix 10% of Phase A data into Phase B
- `N_TRAIN_EXAMPLES = 4000`: Number of training examples per task
- `N_VAL_EXAMPLES = 400`: Number of validation examples per task
- `TASK_A_EPOCHS = 2`: Training epochs for Phase A (MATH)
- `TASK_B_EPOCHS = 2`: Training epochs for Phase B (HotpotQA)
- `RESULTS_DIR = "./drive/MyDrive/"`: Output directory for checkpoints and results

**Output Files:**
- JSON file with training history: `forgetting_history_MATH_to_HotpotQA_{method}_{percentage}pct_bf16.json`
- Individual plot: `sequential_forgetting_curve_MATH_to_HotpotQA_{method}_{percentage}pct_bf16.png`
- Model checkpoint (Phase A): `math_adapter_llama_fp32/`

### `plot_forgetting_history.py`

A plotting utility that combines multiple experiment results into comparison plots.

**Features:**
- Accepts multiple JSON files as input
- Automatically detects metrics (hotpot_loss, math_loss)
- Creates side-by-side subplots for different metrics
- Uses the "name" field from JSON for legend labels
- Supports glob patterns for file selection

**Usage:**

```bash
# Plot specific files
python plot_forgetting_history.py file1.json file2.json file3.json

# Plot with wildcard pattern
python plot_forgetting_history.py drive/MyDrive/forgetting_history_*.json

# Save to specific output file
python plot_forgetting_history.py -o output.png file1.json file2.json

# Combine specific and wildcard
python plot_forgetting_history.py -o comparison.png \
  drive/MyDrive/forgetting_history_*_hard_*.json \
  drive/MyDrive/forgetting_history_*_easy_*.json
```

**Arguments:**
- `json_files`: One or more JSON files (supports glob patterns)
- `-o, --output`: Output file path (default: displays plot or saves to `forgetting_history_plot.png`)

## Understanding the Results

### Metrics Tracked

1. **HotpotQA Validation Loss** (Phase B task)
   - Should decrease during Phase B training (model is learning)
   - Indicates performance on the current task

2. **MATH Validation Loss** (Phase A task)
   - May increase during Phase B training (catastrophic forgetting)
   - Lower increase = better retention of Phase A knowledge
   - This is the key metric for evaluating forgetting mitigation

### Expected Outcomes

Different mixing strategies aim to minimize MATH loss increase while maintaining HotpotQA performance:

- **Difficult problems**: May help retain complex reasoning skills
- **Easy problems**: May provide stable baseline knowledge
- **High-loss problems**: Focus on what model struggles with
- **Low-loss problems**: Reinforce what model learned well
- **Random**: Baseline comparison

## Technical Details

### Model Architecture
- Base model: TinyLlama-1.1B-Chat-v1.0
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- Precision: BFloat16
- LoRA parameters: r=8, alpha=16, dropout=0.05

### Hardware Requirements
- GPU with at least 11GB VRAM
- CUDA support required
- Uses gradient checkpointing to reduce memory usage

### Datasets
- **MATH**: Mathematical problem-solving dataset with human difficulty ratings
  - Dataset: `qwedsacf/competition_math`
  - Has "level" field for difficulty-based selection
- **HotpotQA**: Multi-hop question answering with context
  - Dataset: `hotpot_qa` (distractor configuration)

### Selection Methods Details

1. **`difficulty_hard` / `difficulty_easy`**:
   - Uses the "level" field from MATH dataset (Level 1-5)
   - Sorts problems by human-annotated difficulty
   - Selects top N hardest/easiest problems

2. **`loss_descending` / `loss_ascending`**:
   - Computes per-sample loss using the Phase A trained model
   - Caches losses to `drive/MyDrive/math_losses_{N}_samples.json`
   - Selects top N highest/lowest loss samples
   - First run computes losses; subsequent runs load from cache

3. **`random`**:
   - Random sampling with fixed seed (42) for reproducibility
   - Serves as baseline comparison

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `MAX_SEQ_LENGTH` (currently 2048)
- Reduce `PER_DEVICE_BS` (currently 64)
- Ensure GPU has at least 11GB VRAM

### Missing Dependencies
```bash
pip install torch transformers datasets peft matplotlib tqdm
```

We also include `requirements.txt` containing all the dependencies for these two scripts.