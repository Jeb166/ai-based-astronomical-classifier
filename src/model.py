from keras.models import Sequential
from keras.layers import Dense, Dropout

def build_model(input_dim: int, n_classes: int):
    dnn = Sequential()
    dnn.add(Dense(9, input_dim=input_dim, activation='relu'))
    dnn.add(Dropout(0.1))

    dnn.add(Dense(9, activation='relu'))
    dnn.add(Dropout(0.1))

    dnn.add(Dense(9, activation='relu'))
    dnn.add(Dropout(0.05))

    dnn.add(Dense(6, activation='relu'))
    dnn.add(Dropout(0.05))

    dnn.add(Dense(6, activation='relu'))
    dnn.add(Dense(6, activation='relu'))

    dnn.add(Dense(n_classes, activation='softmax', name='output'))
    dnn.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy']
    )
    return dnn