

"""
when run extract results and pass resutls into a prompt

"""
import os
import shutil
import pandas as pd

from modules.get_combined_logs import get_combined_logs


class GenerateModel():

    def __init__(self, 
                 MODEL_LOGS_PATH=False, 
                 MODEL_SAVE_PATH=False,
                 MODEL_PATH=False,
                 MODEL_DATA_PATH=False,
                 HARD_RESET=False
        ):
        print("Starting model generation")
        if not MODEL_LOGS_PATH or not MODEL_SAVE_PATH or not MODEL_PATH or not MODEL_DATA_PATH:
            raise Exception("Class values not set")
        self.MODEL_LOGS_PATH = MODEL_LOGS_PATH
        self.MODEL_SAVE_PATH = MODEL_SAVE_PATH
        self.MODEL_PATH = MODEL_PATH
        self.MODEL_DATA_PATH = MODEL_DATA_PATH
        self.HARD_RESET = HARD_RESET


    def clean_results(self):
        model_data_df = pd.read_csv(self.MODEL_DATA_PATH, index_col=0)
        model_generation_values = model_data_df["model_generation"].values.tolist()
        model_genreation_values_str = []
        for e in model_generation_values:
            if e < 10:
                e = "0" + str(e)
            else:
                e = str(e)
            model_genreation_values_str.append(e)

        # create combined file array
        combined_file_arr = []
        [combined_file_arr.append(self.MODEL_LOGS_PATH + e) for e in os.listdir(self.MODEL_LOGS_PATH)]
        [combined_file_arr.append(self.MODEL_SAVE_PATH + e) for e in os.listdir(self.MODEL_SAVE_PATH)]
        if self.HARD_RESET: [combined_file_arr.append(self.MODEL_PATH + e) for e in os.listdir(self.MODEL_PATH)]

        for model_path_element in combined_file_arr:
            model_path_element_file = model_path_element.split("/")[-1]
            if model_path_element_file.startswith('model_'):
                file_number = model_path_element_file.split("_")[1].split(".")[0]
                if file_number not in model_genreation_values_str:

                    if os.path.isfile(model_path_element): 
                        os.remove(model_path_element) 
                    else: 
                        shutil.rmtree(model_path_element)
                    print(f"Removed: {model_path_element}")
            



    def get_model_code(self, model_value):
        # get the model code from the model path
        model_code = open(f"{self.MODEL_PATH}model_{model_value}.py", "r").read()
        return model_code

    def get_model_info(self):
            # get the logs from each model
            model_info = ""
            for i in range(1, len(os.listdir(MODEL_LOGS_PATH)) + 1):
                if i < 10:
                    i = "0" + str(i)
                model_code = self.get_model_code(i)
                model_logs = open(f"{MODEL_LOGS_PATH}model_{i}.txt", "r").read()
                model_info += ("MODEL NUMBER: " + i + "\n" + model_code + "\n" + model_logs + "\n\n")

            

            

if __name__ == "__main__":
    MODEL_LOGS_PATH = "/results/model_logs/"
    MODEL_SAVE_PATH = "/results/model_saves/"
    MODEL_PATH = "/models/"
    MODEL_DATA_PATH = "/results/model_data.csv"
    
    HARD_RESET = False

    instance = GenerateModel(
        MODEL_LOGS_PATH=MODEL_LOGS_PATH, 
        MODEL_SAVE_PATH=MODEL_SAVE_PATH,
        MODEL_PATH=MODEL_PATH,
        MODEL_DATA_PATH=MODEL_DATA_PATH,
        HARD_RESET=HARD_RESET
    )
    instance.get_model_info()
    instance.clean_results()