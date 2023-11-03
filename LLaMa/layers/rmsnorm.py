import tensorflow as tf

class RMSNorm(tf.keras.layers.Layer):
  def __init__(self, epsilon = 0.001):
    super(RMSNorm, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):

    self.units = input_shape[-1]

    self.alpha = self.add_weight(
              shape = (self.units,),
              initializer = 'ones',
              trainable = True,
              dtype = tf.float32
    )

    self.beta = self.add_weight(
              shape = (self.units,),
              initializer = 'zeros',
              trainable = True,
              dtype = tf.float32
    )

  def call(self, inputs):
    RMS_a = tf.math.sqrt(1/self.units * tf.math.reduce_sum(inputs**2, axis = -1, keepdims = True)) + self.epsilon
    return inputs / RMS_a * self.alpha + self.beta