import os
from typing import Tuple

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_dataset_generators(data_dir, batch_size, target_size:Tuple[int, int]=(150, 150)):
    """Create the Keras image dataset generators"""
    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "validation")

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       fill_mode='nearest',
                                       horizontal_flip=True,)
    
    val_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      fill_mode='nearest',
                                      horizontal_flip=True,)

    # Flow training images in batches of batch_size using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=target_size,  # All images will be resized to the target_size
        batch_size=batch_size,
        class_mode="categorical",
    )

    # Flow validation images in batches of batch_size using test_datagen generator
    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    return train_generator, validation_generator
