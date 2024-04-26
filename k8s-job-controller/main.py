import pandas as pd
import os


"""

Check for new model files

If need for a new model start the model creation application via k8s

If there are modeles that are ready for training and free resources start the training process

"""

MODEL_DIR = os.environ["MODEL_DIR"]
RESULT_DIR = os.environ["RESULT_DIR"]
NUMBER_OF_MODELS = os.environ["NUMBER_OF_MODELS"]

def main():

    print("MODEL_DIR:", MODEL_DIR, "\nRESULT_DIR:", RESULT_DIR, "\nNUMBER_OF_MODELS:", NUMBER_OF_MODELS)



    # Check for new model files
    


    pass

if __name__ == "__main__":
    main()

