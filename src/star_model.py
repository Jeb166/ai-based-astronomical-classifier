from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU

def build_star_model(input_dim: int, n_classes: int):
    model = Sequential([
        Dense(64, input_shape=(input_dim,)),
        LeakyReLU(alpha=0.05),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32),
        LeakyReLU(alpha=0.05),
        BatchNormalization(),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model
