import os
import pandas as pd

def get_model_generation():
    files = os.listdir("./models")

    for file in files:
        if not "model" in file or not ".py" in file:
            continue
        model_generation = int(file.split(".")[0].split("_")[1])
        return model_generation

def check_completed_models(model_generation):
    model_result_df = pd.read_csv("./results/model_data.csv")

    print(model_result_df.head())



def get_model():
    model_generation = get_model_generation()
    check_completed_models(model_generation)

    return


