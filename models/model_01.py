import tensorflow as tf

def model():

    learning_rate = 0.00001

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3095,)),
        tf.keras.layers.Dense(50, activation="leaky_relu"),
        tf.keras.layers.Dense(10, activation="leaky_relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(
        learning_rate),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
        ]
    )

    return model