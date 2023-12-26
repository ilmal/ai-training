import tensorflow as tf
import pandas as pd
import time

class CustomCallback(tf.keras.callbacks.Callback):
    # https://www.tensorflow.org/guide/keras/writing_your_own_callbacks#a_basic_example

    def __init__(self, model_value, model_save_path, model_logs_save_path, model_dataframe_path):
        self.model_value = model_value
        self.model_save_path = model_save_path
        self.model_logs_save_path = model_logs_save_path
        self.model_dataframe_path = model_dataframe_path
        # print(model_value)
        # print(type(model_value))

    def on_train_begin(self, logs=None):
        print("Starting training...")
        # create the log file
        self.log_file = open(f"{self.model_logs_save_path}model_{self.model_value}.txt", "w")
        self.log_file.close()

        df = pd.read_csv(self.model_dataframe_path, index_col=0)

        df = pd.concat([df, pd.DataFrame([{
            'model_generation': self.model_value, 
            'training_finished': False
        }])], ignore_index=True)

        df.to_csv(self.model_dataframe_path)
        

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if batch % 10 == 0:
            # print(f"Batch {batch}/{self.params['steps']} - {logs}",flush=True, end="\n")
            pass

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}/{self.params['epochs']} - {logs}", flush=True, end="\n")
        self.log_file = open(f"{self.model_logs_save_path}model_{self.model_value}.txt", "a")
        self.log_file.write(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs.get('loss'):.8f}, accuracy: {logs.get('accuracy'):.5f}, val_loss: {logs.get('val_loss'):.8f}, val_accuracy: {logs.get('val_accuracy'):.5f}\n")
        self.log_file.close()

    def on_train_end(self, logs=None):
        print("Training finished") 
        df = pd.read_csv(self.model_dataframe_path, index_col=0)

        df.loc[df['model_generation'] == int(self.model_value), 'training_finished'] = True

        df.to_csv(self.model_dataframe_path)


        