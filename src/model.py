from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    LeakyReLU
)
from keras.regularizers import l2


def build_model(input_dim: int, n_classes: int):
    
    model = Sequential()

    model.add(
        Dense(
            128,
            input_dim=input_dim,
            kernel_regularizer=l2(1e-4)
        )
    )
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(
        Dense(
            64,
            kernel_regularizer=l2(1e-4)
        )
    )
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(
        Dense(
            32,
            kernel_regularizer=l2(1e-4)
        )
    )
    model.add(LeakyReLU(alpha=0.05))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(n_classes, activation="softmax", name="output"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model
