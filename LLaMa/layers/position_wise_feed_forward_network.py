import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
import SwiGLU

def ffn(d_ff = 1365, d_model = 512):
    # TODO: Update document
    return Sequential([
        SwiGLU(d_ff),
        Dense(d_model)
    ])