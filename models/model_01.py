import tensorflow as tf

def model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3095,)),
        tf.keras.layers.Dense(200, activation="leaky_relu"),
        tf.keras.layers.Dense(100, activation="leaky_relu"),
        tf.keras.layers.Dense(100, activation="leaky_relu"),
        tf.keras.layers.Dense(100, activation="leaky_relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0000001),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
        ]
    )

    return model