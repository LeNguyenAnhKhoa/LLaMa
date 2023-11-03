import tensorflow as tf

class SwiGLU(tf.keras.layers.Layer):
  def __init__(self, units):
    super(SwiGLU, self).__init__()
    self.units = units

  def build(self, input_shape):

    self.W1 = self.add_weight(
              shape = (input_shape[-1], self.units),
              initializer = 'random_normal',
              trainable = True,
              dtype = tf.float32
    )
        
    self.b1 = self.add_weight(
              shape = (self.units,),
              initializer = 'zeros',
              trainable = True,
              dtype = tf.float32
    )

    self.W2 = self.add_weight(
              shape = (input_shape[-1], self.units),
              initializer = 'random_normal',
              trainable = True,
              dtype = tf.float32
    )
        
    self.b2 = self.add_weight(
              shape = (self.units,),
              initializer = 'zeros',
              trainable = True,
              dtype = tf.float32
    )

  def call(self, inputs):
    v1 = tf.keras.activations.swish(tf.matmul(inputs, self.W1) + self.b1)
    v2 = tf.matmul(inputs, self.W2) + self.b2
    return v1 * v2