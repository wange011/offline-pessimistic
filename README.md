# Sample Code for "Pessimism Meets Invariance: Provably Efficient Offline Mean-Field Multi-Agent RL"

This is the official codebase for Pessimism Meets Invariance: Provably Efficient Offline Mean-Field Multi-Agent RL. Here, we provide a sample implementation of SAFARI on the cooperative navigation environment. This specific repository is untested; however, many of the given files match the code used to run experiments in the paper exactly. Refer to `agents/safari.py`.

## Requirements

To install requirements, run:

```pip install -r requirements.txt```

Not all dependencies may be used; however, all dependencies that are needed can be found here.

## Run

To kick off a training run of SAFARI, add a dataset into the `data/<scenario_name>` folder. Then running:

```python main.py safari```

will start the script from the entry point, `main.py`.

## Data Format

SAFARI expects there to be a dataset present at `data/<scenario_name>/<idx_seed>` for each parallel seed that is run. We expect three files:

1. actions.txt (Shape: [N, H])
2. rewards.txt (Shape: [N, H])
3. obs.txt (Shape: [N, H, O])

each of which expects each line to be an episodic trajectory. We convert each buffer into a list (1), cast them to `str` (2), and print them on separate lines of the file (3). 
