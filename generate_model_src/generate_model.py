

"""
when run extract results and pass resutls into a prompt

"""
import os

from modules.get_combined_logs import get_combined_logs


class GenerateModel():

    def __init__(self, 
                 LOGS_PATH=False, 
                 MODEL_PATH=False
        ):
        if not LOGS_PATH or not MODEL_PATH:
            raise Exception("Class values not set")
        self.LOGS_PATH = LOGS_PATH
        self.MODEL_PATH = MODEL_PATH
        
    def get_model_code(self, model_value):
        # get the model code from the model path
        model_code = open(f"{self.MODEL_PATH}model_{model_value}.py", "r").read()
        return model_code

    def get_model_info(self):
            # get the logs from each model
            model_info = ""
            for i in range(1, len(os.listdir(LOGS_PATH)) + 1):
                if i < 10:
                    i = "0" + str(i)
                model_code = self.get_model_code(i)
                model_logs = open(f"{LOGS_PATH}model_{i}.txt", "r").read()
                model_info += ("MODEL NUMBER: " + i + "\n" + model_code + "\n" + model_logs + "\n\n")

            




if __name__ == "__main__":

    LOGS_PATH = "/results/model_logs/"
    MODEL_PATH = "/models/"

    instance = GenerateModel(
        LOGS_PATH=LOGS_PATH, 
        MODEL_PATH=MODEL_PATH
    )
    instance.get_model_info()