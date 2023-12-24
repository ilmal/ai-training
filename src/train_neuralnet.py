import os
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = "-1"

import tensorflow as tf
from IPython.display import clear_output

# Module imports
from modules.data_generator import data_generator
from modules.get_model import get_model
from modules.custom_callback import CustomCallback
# from modules.training import ...

"""
Plan:

-- When new models are found run the training algorithem by getting the model from a folder containing models
-- Save the model values to folder and save results to csv file
-- use results to generate new models using bard API
-- repeat
-- profit :)

"""

# check for GPU
print("\n")
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)

        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU found on device")
print("\n")

class Train_neuralnet:
    def __init__(self, val_data_list=False, data_list=False, val_data_dir=False, data_dir=False, BATCH_SIZE=False, CHECKPOINT_PATH=False):
        if not BATCH_SIZE or not CHECKPOINT_PATH:
            raise Exception("BATCH_SIZE or CHECKPOINT_PATH not set")
        self.data_list = data_list
        self.val_data_list = val_data_list
        self.val_data_dir = val_data_dir
        self.data_dir = data_dir
        self.BATCH_SIZE = BATCH_SIZE
        self.CACHE_PATH = self.get_cache_path(BATCH_SIZE, "train")
        self.VAL_CACHE_PATH = self.get_cache_path(BATCH_SIZE, "val")
        self.CHECKPOINT_PATH = CHECKPOINT_PATH
    

    def get_model(self):
        self.model_value, self.model = get_model()

    def get_cache_path(self, BATCH_SIZE, cahe_type):
        base_folder = "/data/"
        caches = os.listdir(base_folder)
        for cache in caches:
            if cahe_type in cache: return base_folder + cache


    def get_data(self):
        output_signature=(
                    tf.TensorSpec(shape=(None, 3095), dtype=tf.float32),
                    tf.TensorSpec(shape=(None), dtype=tf.int64)
        )
        if self.val_data_list and self.data_list and self.val_data_dir and self.data_dir:
            generator = data_generator(self.data_dir, self.BATCH_SIZE)
            val_generator = data_generator(self.val_data_dir, self.BATCH_SIZE)

            val_generator_dataset = tf.data.Dataset.from_generator(
                lambda: val_generator, output_signature)

            generator_dataset = tf.data.Dataset.from_generator(
                lambda: generator, output_signature)
        else:
            dum_gen = lambda : None 
            val_generator_dataset = tf.data.Dataset.from_generator(dum_gen,output_signature=output_signature)
            generator_dataset = tf.data.Dataset.from_generator(dum_gen,output_signature=output_signature)

        print("VAL_CACHE_PATH", self.VAL_CACHE_PATH)
        print("TRAIN_CACHE_PATH", self.CACHE_PATH)
        print(os.listdir(self.CACHE_PATH))

        for i, e in enumerate(val_generator_dataset):
            if i > 3: break
            print(e)

        self.val_generator_dataset = val_generator_dataset.cache(self.VAL_CACHE_PATH + "/tf_cache.tfcache").shuffle(100)

        self.generator_dataset = generator_dataset.cache(self.CACHE_PATH + "/tf_cache.tfcache").shuffle(100)

        self.generator_dataset = generator_dataset.prefetch(tf.data.AUTOTUNE)
        self.val_generator_dataset = val_generator_dataset.prefetch(tf.data.AUTOTUNE)

    def callbacks(self, SAVE_INTERVAL = 2, LOGS_DIR = "data/logs"):
        # steps_per_epoch = len(self.data_list)
        steps_per_epoch = 10000
    
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
           filepath=CHECKPOINT_PATH,
           save_weights_only=False,
           save_freq=int(steps_per_epoch * SAVE_INTERVAL))

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard( 
            log_dir=(LOGS_DIR),
            histogram_freq=1, 
            write_steps_per_second=True
        )

    def train(self):
        self.model.summary()

        self.model.fit(self.generator_dataset,
                validation_data=self.val_generator_dataset,
                epochs=10,
                callbacks=[
                    self.tensorboard_callback, 
                    self.model_checkpoint_callback, 
                    CustomCallback(self.model_value),    
                ],
                initial_epoch=0,
                verbose=1,
                
        )
        print("saving model")
        self.model.save("./saved_model_new/model")


if __name__ == "__main__":
    BATCH_SIZE = 10

    CHECKPOINT_PATH = "checkpoints/"

    # runtime = Main(val_data_list, data_list, val_data_dir, data_dir, BATCH_SIZE, CHECKPOINT_PATH)
    runtime = Train_neuralnet(BATCH_SIZE=BATCH_SIZE, CHECKPOINT_PATH=CHECKPOINT_PATH)
    runtime.get_model()
    runtime.get_data()
    runtime.callbacks()

    runtime.train()

