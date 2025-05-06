import pandas as pd, numpy as np, tensorflow as tf
from datamodule import _add_colour_features, CLASS_MAP

def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path, compile=False)

def row_to_tensor(row: pd.Series) -> np.ndarray:
    row = _add_colour_features(row.to_frame().T)
    return row[["u","g","r","i","z","u_g","g_r","r_i","i_z"]].values.astype(np.float32)

def predict_row(model: tf.keras.Model, row: pd.Series):
    x = row_to_tensor(row)
    probs = model.predict(x, verbose=0)[0]
    inv   = {v:k for k,v in CLASS_MAP.items()}
    idx   = probs.argmax()
    return inv[idx], float(probs[idx])
