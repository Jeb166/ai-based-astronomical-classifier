from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input

def build_star_model(input_dim: int, n_classes: int):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.4),  # Biraz artırdım
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(), 
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),  # Ek katman
        BatchNormalization(),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model
