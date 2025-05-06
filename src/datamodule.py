import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLASS_MAP = {"GALAXY": 0, "STAR": 1, "QSO": 2}

def _add_colour_features(df: pd.DataFrame) -> pd.DataFrame:
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        df[f"{a}_{b}"] = df[a] - df[b]
    return df

def load_data(csv_path: str,
              test_size: float = 0.15,
              val_size: float  = 0.15,
              seed: int        = 42):
    """Return ((Xt,yt),(Xv,yv),(Xtst,ytst), n_features)."""
    df = pd.read_csv(csv_path, encoding="utf‑8")
    df = _add_colour_features(df)

    X = df[["u","g","r","i","z","u_g","g_r","r_i","i_z"]].values.astype(np.float32)
    y = df["class"].map(CLASS_MAP).values.astype(np.int64)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=val_size+test_size,
        stratify=y, random_state=seed)

    rel_val = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_val,
        stratify=y_tmp, random_state=seed)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    return (to_t(X_train), torch.tensor(y_train)), \
           (to_t(X_val),   torch.tensor(y_val)),   \
           (to_t(X_test),  torch.tensor(y_test)),  \
           X_train.shape[1]
