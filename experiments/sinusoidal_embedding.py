"""
THIS SCRIPT CONTAINS A CUSTOM SINUSOIDAL 
EMBEDDING FOR THE u_net.py script
"""

""" ALL IMPORTS """
import math
import tensorflow as tf
import keras
from keras import layers

"""
Creating a custom layer called Sinusoidal Embedding
to replace Lambda layers that were causing  errors 
(probably due to dependencies reasons)
"""
@keras.saving.register_keras_serializable()
class SinusoidalEmbedding(layers.Layer): 
    def __init__(self, embedding_dims, **kwargs):
        super(SinusoidalEmbedding, self).__init__(**kwargs) # sets up the layer by calling the superclass
        self.embedding_dims = embedding_dims
    def build(self, input_shape):
        """
        Taken from the local sinusoidal_embedding 
        Build method is called when building the layer
        """
        self.embedding_min_frequency = 1.0
        self.embedding_max_frequency = 1000.0  # You can adjust this value
        frequencies = tf.exp(
            tf.linspace(
                tf.math.log(self.embedding_min_frequency),
                tf.math.log(self.embedding_max_frequency),
                self.embedding_dims // 2,
            )
        )
        angular_speeds = 2.0 * tf.constant(math.pi) * frequencies
        self.angular_speeds = tf.cast(angular_speeds, dtype=tf.float32)
        """
        We compute the frequencies for the sinusoidal embeddings 
        using exponential and logarithmic operations.
        """
    def call(self, x):
        # Assuming x is a scalar (e.g., noise_variances)
        # call layer defines how the layer processes the input data
        embeddings = tf.concat(
            [tf.sin(self.angular_speeds * x), tf.cos(self.angular_speeds * x)], axis=-1
        )
        return embeddings
        """
        We compute the sinusoidal embeddings by concatenating sine
        and cosine functions of the angular speeds.
        The output embeddings contain both sine and cosine components.
        """