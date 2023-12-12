import os
import tensorflow as tf

# Module imports
from modules.data_generator import data_generator
from modules.get_model import get_model
# from modules.training import ...

"""
Plan:

-- When new models are found run the training algorithem by getting the model from a folder containing models
-- Save the model values to folder and save results to csv file
-- use results to generate new models using bard API
-- repeat
-- profit :)

"""



class Main:
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
        self.model = get_model()

    def get_cache_path(self, BATCH_SIZE, type):
        return
        base_folder = "./data/cache/"
        caches = os.listdir(base_folder)
        for cache in caches:
            if str(BATCH_SIZE) in cache:
                return f"{base_folder}{type}"

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
        
        # self.val_generator_dataset = val_generator_dataset.cache(VAL_CACHE_PATH + "tf_cache.tfcache").shuffle(100)

        # self.generator_dataset = generator_dataset.cache(CACHE_PATH + "tf_cache.tfcache").shuffle(100)

        self.generator_dataset = generator_dataset.prefetch(tf.data.AUTOTUNE)

    def callbacks(self, SAVE_INTERVAL = 2, LOGS_DIR = "data/logs"):
        steps_per_epoch = len(self.data_list)
    
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
           filepath=CHECKPOINT_PATH,
           save_weights_only=False,
           save_freq=int(steps_per_epoch * SAVE_INTERVAL))

        self.tensorboard_callback = tf.keras.callbacks.TensorBoard( 
            log_dir=(LOGS_DIR),
            histogram_freq=1, 
            write_steps_per_second=True
        )

    def train():
        self.model.summary()
        pass

# val_data_list = os.listdir("/mnt/external/app/AI-project/get_training_data_app/dev/output_val/labels/")
# data_list = os.listdir("/mnt/external/app/AI-project/get_training_data_app/dev/output_train/labels/")

# val_data_dir = "/mnt/external/app/AI-project/get_training_data_app/dev/output_val/"
# data_dir = "/mnt/external/app/AI-project/get_training_data_app/dev/output_train/"

CACHE_PATH = "./cache/"
VAL_CACHE_PATH = "./cache_val/"

BATCH_SIZE = 1000

CHECKPOINT_PATH = "checkpoints/"

# runtime = Main(val_data_list, data_list, val_data_dir, data_dir, BATCH_SIZE, CHECKPOINT_PATH)
runtime = Main(BATCH_SIZE=BATCH_SIZE, CHECKPOINT_PATH=CHECKPOINT_PATH)
runtime.get_model()
runtime.get_data()
runtime.callbacks()

runtime.train()

