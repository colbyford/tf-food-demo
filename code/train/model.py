from typing import Tuple

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.optimizers import RMSprop, Adam, Adagrad, Adadelta

OPTIMIZERS = {
    'adam': Adam,
    'rmsprop': RMSprop,
    'adagrad': Adagrad,
    'adadelta': Adadelta
}

def create_model(lr, classes=1, optimizer_name='rmsprop', target_size:Tuple[int, int]=(150, 150)):
    """Generate TensorFlow Model Object"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(*target_size, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # 
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flattened prediction image
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(classes, activation="softmax"),
    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer=OPTIMIZERS[optimizer_name](lr=lr),
                  metrics=["acc"])

    return model

def load_transfer_model(lr, classes=1, optimizer=RMSprop, target_size:Tuple[int, int]=(150, 150)):

    base_model = InceptionV3(input_shape=(*target_size, 3),
                             include_top=False,
                             weights='imagenet')
    
    

