source .venv/bin/activate && python difficulty_experiments/difficulty_mixing.py --selection-method "difficulty_hard"
source .venv/bin/activate && python difficulty_experiments/difficulty_mixing.py --selection-method "random"
source .venv/bin/activate && python difficulty_experiments/difficulty_mixing.py --selection-method "difficulty_easy"
source .venv/bin/activate && python difficulty_experiments/difficulty_mixing.py --selection-method "loss_ascending"

# Then run the following script, replacing the placeholder with the output files
# source .venv/bin/activate && python difficulty_mixing/plot_forgetting_history.py -o test_output.png # <put output files here> 2>&1