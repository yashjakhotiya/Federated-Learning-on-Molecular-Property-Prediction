import wandb
import numpy as np

from collections import defaultdict
from tqdm import tqdm

ENTITY = 'sysml-proj-team-13'
PROJECT = 'fedml_test'
METRIC_NAME = 'AUCROC'

api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}")

for run in tqdm(runs):
    if METRIC_NAME not in run.history().columns:
        continue
    values = run.history()[METRIC_NAME].values
    values = values[~np.isnan(values)]

    run.summary[f"{METRIC_NAME}_max"] = np.max(values)
    run.summary[f"{METRIC_NAME}_min"] = np.min(values)
    run.summary[f"{METRIC_NAME}_std"] = np.std(values)

    run.summary.update()