import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical

def load_and_prepare(filename: str):
    # Read and shuffle data
    sdss_df = pd.read_csv(filename, encoding='utf-8')
    sdss_df = sdss_df.sample(frac=1)

    # Drop physically insignificant columns
    sdss_df = sdss_df.drop(
        ['objid', 'specobjid', 'run', 'rerun', 'camcol', 'field'],
        axis=1
    )

    # Partition SDSS data (60% train, 20% validation, 20% test)
    train_count = 60000
    val_count = 20000
    test_count = 20000

    train_df = sdss_df.iloc[:train_count]
    validation_df = sdss_df.iloc[train_count:train_count+val_count]
    test_df = sdss_df.iloc[-test_count:]

    # Extract features
    X_train = train_df.drop(['class'], axis=1)
    X_validation = validation_df.drop(['class'], axis=1)
    X_test = test_df.drop(['class'], axis=1)

    # One-hot encode labels
    le = LabelEncoder()
    le.fit(sdss_df['class'])
    encoded_Y = le.transform(sdss_df['class'])
    onehot_labels = to_categorical(encoded_Y)

    y_train = onehot_labels[:train_count]
    y_validation = onehot_labels[train_count:train_count+val_count]
    y_test = onehot_labels[-test_count:]

    # Scale features (fit on train only)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_validation, X_test, y_train, y_validation, y_test, sdss_df
