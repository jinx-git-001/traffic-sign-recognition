import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_model():

    input_layer = layers.Input(shape=(32, 32, 1))

    conv1 = layers.Conv2D(
        32,
        (5,5),
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0001)
    )(input_layer)

    pool1 = layers.MaxPooling2D((2,2))(conv1)
    drop1 = layers.Dropout(0.1)(pool1)

    conv2 = layers.Conv2D(
        64,
        (5,5),
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0001)
    )(drop1)

    pool2 = layers.MaxPooling2D((2,2))(conv2)
    drop2 = layers.Dropout(0.2)(pool2)

    conv3 = layers.Conv2D(
        128,
        (5,5),
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0001)
    )(drop2)

    pool3 = layers.MaxPooling2D((2,2))(conv3)
    drop3 = layers.Dropout(0.3)(pool3)

    flat = layers.Flatten()(drop3)

    dense = layers.Dense(1024, activation='relu')(flat)

    drop4 = layers.Dropout(0.5)(dense)

    output = layers.Dense(43, activation='softmax')(drop4)

    model = models.Model(inputs=input_layer, outputs=output)

    return model