import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):
    # https://www.tensorflow.org/guide/keras/writing_your_own_callbacks#a_basic_example

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if batch % 10 == 0:
            # print(f"Batch {batch}/{self.params['steps']} - {logs}",flush=True, end="\n")
            pass