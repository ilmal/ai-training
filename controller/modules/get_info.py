import pandas as pd
import os
import numpy as np

def get_trainable_models(RESULTS_DIR, MODEL_DIR):

    if not os.path.exists(RESULTS_DIR + "model_data.csv"): return False

    df = pd.read_csv(RESULTS_DIR + "model_data.csv")

    models_in_training = [int(e) for e in df["model_generation"].tolist()]
    models_ready_for_training = [int(e.split("_")[1].split(".")[0]) for e in os.listdir(MODEL_DIR) if "model_" in e]

    # get models that are ready for training and not in training
    diff = list(set(models_ready_for_training) - set(models_in_training))

    return diff