import os
import inspect
import sys

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from azureml.core import Run
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Append the current file path to the system PATH to allow the system to find the submodules
current_file_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(current_file_path)

from aml_callback import AzureMLCallback
from data_prep import create_dataset_generators
from model import create_model

@click.command()
@click.option("--data-dir", type=str, default=os.environ.get("AZUREML_DATAREFERENCE_food_images"),
              help="The directory where the data is stored.",)
@click.option("-e", "--epochs", default=5, type=int,
              help="The number of epochs to train the neural network")
@click.option("-m", "--minibatch-size", "batch_size", default=32, type=int,
              help="The number of images in each minibatch",)
@click.option("-l", "--learning-rate", "lr", default=1e-3, type=float,
              help="The learning rate for the algorithm")
@click.option("-o", "--optimizer", type=click.Choice(['adadelta', 'rmsprop', 'adagrad', 'adam']), default='rmsprop',
              help='The optimizer to use for training the model')
@click.argument("categories", nargs=-1, type=str)
def train(data_dir, epochs, batch_size, lr, optimizer, categories):
    """Train the neural network"""
    run = Run.get_context()

    run.log('optimizer', optimizer)
    run.log('minibatch_size', batch_size)
    run.log('learning_rate', lr)
    run.log('categories', categories)

    # Get model and data objects
    train_generator, validation_generator = create_dataset_generators(
        data_dir, batch_size, categories
    )
    
    model = create_model(lr=lr, classes=train_generator.num_classes, optimizer_name=optimizer)
    print(model.optimizer)
    
    os.makedirs("./outputs", exist_ok=True)

    aml_callback = AzureMLCallback(run)
    checkpointer = ModelCheckpoint(
        filepath="./outputs/weights_{epoch:02d}.hdf5", period=25)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples / batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / batch_size,
        verbose=2,
        callbacks=[aml_callback, checkpointer],
    )

    model.save("./outputs/final_model.hdf5")


if __name__ == "__main__":
    train()
