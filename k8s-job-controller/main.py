from modules.get_info import get_trainable_models
from modules.docker_controller import get_active_jobs, start_training

"""

Check for new model files

If need for a new model start the model creation application via k8s

If there are modeles that are ready for training and free resources start the training process

"""

# MODEL_DIR = os.environ["MODEL_DIR"]
# RESULT_DIR = os.environ["RESULT_DIR"]
# NUMBER_OF_MODELS = os.environ["NUMBER_OF_MODELS"]

def main():

    MODEL_DIR = "../models/" # This is the directory where the model code are stored
    RESULT_DIR = "../results/" 
    NUMBER_OF_MODELS = 3
    CONCURRENT_JOBS = 1 # This is the number of models that can be trained at the same time
    IMAGE_NAME = "ai-training" 

    # Check for new model files
    trainable_models = get_trainable_models(RESULT_DIR, MODEL_DIR)
    if not trainable_models: 
        raise Exception("Problem getting data from results")

    if len(trainable_models) == 0: 
        print("No new models to train")
        return
    
    # Start training process with new models

    # Check for free resources
    active_jobs = get_active_jobs()

    if len(active_jobs) >= CONCURRENT_JOBS:
        print("No free resources")
        return

    # Start training process
    start_training(IMAGE_NAME, trainable_models[0])

    pass

if __name__ == "__main__":
    main()