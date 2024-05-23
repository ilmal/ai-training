

"""
when run extract results and pass resutls into a prompt

"""
import os
import shutil
import pandas as pd
import json
import requests
import textwrap

from modules.get_combined_logs import get_combined_logs


class GenerateModel():

    def __init__(self, 
                 MODEL_LOGS_PATH=False, 
                 MODEL_SAVE_PATH=False,
                 MODEL_PATH=False,
                 MODEL_DATA_PATH=False,
                 HARD_RESET=False,
                 BARD_API=False,
                 BARD_SESSION_ID=False,
        ):
        print("Starting model generation")
        if not MODEL_LOGS_PATH or not MODEL_SAVE_PATH or not MODEL_PATH or not MODEL_DATA_PATH or not BARD_API or not BARD_SESSION_ID:
            raise Exception("Class values not set")
        self.MODEL_LOGS_PATH = MODEL_LOGS_PATH
        self.MODEL_SAVE_PATH = MODEL_SAVE_PATH
        self.MODEL_PATH = MODEL_PATH
        self.MODEL_DATA_PATH = MODEL_DATA_PATH
        self.HARD_RESET = HARD_RESET
        self.BARD_API = BARD_API
        self.BARD_SESSION_ID = BARD_SESSION_ID


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
                model_info += ("MODEL CODE FOR MODEL: " + i + "\n" + model_code + "\nAND THE RESULT\n" + model_logs + "\n\n")
                self.model_info = model_info

    def create_prompt(self):
        self.prompt = f"""
Please help me build a new model. The model should follow this structure and must be able to run like this code:

import tensorflow as tf

def model():

    learning_rate = <set learning rate>

    model = tf.keras.Sequential([
        <define the new model>
    ])

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
        ]
    )

    return model


These are some of my previous models and their results, please use this information to build an even better model and output the code.
I want your output to only be the code I need to run, Its very important that you say nothing but the code, only the code. If you answer me with anything
but just the code it will not work. Please analyse my previous results and output only the code for a new and improved code for me to run.

{self.model_info}
"""
        return self.prompt

    def get_model_name(self):
        model_df = pd.read_csv(MODEL_DATA_PATH, index_col=0)

        new_model_number = model_df["model_generation"].values.tolist()[-1] + 1

        if new_model_number < 10:
            new_model_number_string = "0" + str(new_model_number)
        else: 
            new_model_number_string = str(new_model_number)

        return "model_" + new_model_number_string + ".py"

    def connect_bard(self):

        payload = json.dumps({
        "session_id": self.BARD_SESSION_ID,
        "message": self.prompt
        })

        headers = {
        'Content-Type': 'text/plain',
        'Content-Type': 'application/json'
        }

        print("Connecting to bard...")
        response = requests.request("POST", self.BARD_API, headers=headers, data=payload)

        json_object = json.loads(response.text)

        response_str = json.dumps(json_object["choices"][0]["content"][0])

        split_point = "```"
        start_index = response_str.find(split_point)
        end_index = response_str.find(split_point, start_index + 1)

        edit_response = response_str[start_index + 8:end_index -1]

        edit_response = edit_response.replace("```", "")
        edit_response = edit_response.replace('\\"', '"')
        edit_response = edit_response[1:]
        edit_response = edit_response[:-1]

        if not "import tensorflow as tf" in edit_response or not "def model():" in edit_response:
            print("The code was not correct, restarting connection.")
            self.connect_bard()

        lines = edit_response.split("\\n")
        file = open(self.MODEL_PATH + self.get_model_name(), "w")
    
        for line in lines:
            # print(line)
            file.write(line + "\n")
        file.close()
        print("New model created successfully!")


            



if __name__ == "__main__":
    MODEL_LOGS_PATH = "/results/model_logs/"
    MODEL_SAVE_PATH = "/results/model_saves/"
    MODEL_PATH = "/models/"
    MODEL_DATA_PATH = "/results/model_data.csv"
    
    HARD_RESET = False
    BARD_API = "http://192.168.1.247:8000/ask/"
    BARD_SESSION_ID = "egi1U5WvNtxqDLQO_aZLl4b1qGOEu9vh-tcs61f3RsIjnlSqaf0XcCzn4Xa10szhjQ513w."

    instance = GenerateModel(
        MODEL_LOGS_PATH=MODEL_LOGS_PATH, 
        MODEL_SAVE_PATH=MODEL_SAVE_PATH,
        MODEL_PATH=MODEL_PATH,
        MODEL_DATA_PATH=MODEL_DATA_PATH,
        HARD_RESET=HARD_RESET,
        BARD_API=BARD_API,
        BARD_SESSION_ID=BARD_SESSION_ID,
    )
    instance.clean_results()
    instance.get_model_info()
    prompt = instance.create_prompt()
    instance.connect_bard()

