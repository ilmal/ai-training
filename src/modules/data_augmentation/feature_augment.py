class FeatureAugment(tf.keras.layers.Layer):
  def __init__(self, num_features, scale_factor):
    super(FeatureAugment, self).__init__()
    self.num_features = num_features
    self.scale_factor = scale_factor

  def call(self, inputs):
    '''
    apply a gradual scale factor to the features, aka more recent data more important

    '''
    # Split the input into two parts: the first num_features and the rest
    x1, x2 = tf.split(inputs, [self.num_features, -1], axis=-1)
    
    def create_scale_factors(input, max_output):
        input_len = len(input)
        


    # Scale the first part by the scale factor
    x1 = x1 * self.scale_factor

    # Concatenate the scaled part and the rest
    outputs = tf.concat([x1, x2], axis=-1)
    return outputs
