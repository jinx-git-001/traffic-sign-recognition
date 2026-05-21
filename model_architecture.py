import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Concatenate
)
from tensorflow.keras.models import Model


def build_model():

    inputs = Input(shape=(32, 32, 1))

    # Block 1
    x1 = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
    x1_pool = MaxPooling2D(pool_size=(2, 2))(x1)
    x1_drop = Dropout(0.25)(x1_pool)

    # Block 2
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same')(x1_drop)
    x2_pool = MaxPooling2D(pool_size=(2, 2))(x2)
    x2_drop = Dropout(0.25)(x2_pool)

    # Block 3
    x3 = Conv2D(128, (5, 5), activation='relu', padding='same')(x2_drop)
    x3_pool = MaxPooling2D(pool_size=(2, 2))(x3)
    x3_drop = Dropout(0.25)(x3_pool)

    # Additional pooled branches
    branch1 = MaxPooling2D(pool_size=(4, 4))(x1_drop)
    branch2 = MaxPooling2D(pool_size=(2, 2))(x2_drop)

    # Flatten all branches
    flat1 = Flatten()(branch1)
    flat2 = Flatten()(branch2)
    flat3 = Flatten()(x3_drop)

    # Concatenate
    merged = Concatenate()([flat1, flat2, flat3])

    # Dense layers
    dense = Dense(1024, activation='relu')(merged)
    dense = Dropout(0.5)(dense)

    outputs = Dense(43, activation='softmax')(dense)

    model = Model(inputs=inputs, outputs=outputs)

    return model