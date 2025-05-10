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

        # --- Color indices (fotometrik farklar) ---
    sdss_df["u_g"] = sdss_df["u"] - sdss_df["g"]
    sdss_df["g_r"] = sdss_df["g"] - sdss_df["r"]
    sdss_df["r_i"] = sdss_df["r"] - sdss_df["i"]
    sdss_df["i_z"] = sdss_df["i"] - sdss_df["z"]


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

def load_star_subset(filename: str):
    df = pd.read_csv(filename, encoding='utf-8')
    star_df = df[df["class"] == "STAR"].copy()

    star_df = star_df.dropna(subset=["subClass"])      # bozukları at

    # ----- 1)  EN AZ 5 ÖRNEĞİ OLAN alt‑türler kalsın  -----------------
    cnt = star_df["subClass"].value_counts()
    star_df = star_df[star_df["subClass"].isin(cnt[cnt >= 5].index)]

    print("Alt‑tür frekansları (>=5 tutuldu):")
    print(star_df["subClass"].value_counts().head(20))


    y = star_df["subClass"]
    X = star_df.drop(["class", "subClass", "objid",
                      "specobjid", "run", "rerun",
                      "camcol", "field"], axis=1)

    # split: 70% train, 15% val, 15% test
    from sklearn.model_selection import train_test_split
    # ---------- 2)  TEK stratify'lı bölme -----------------------------
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)   # stratify = y  ✓

    # ikinci bölmede stratify KAPALI  →  nadir sınıflar 0 hatası yok
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=None)


    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    le = LabelEncoder().fit(y_train)
    y_train_oh = to_categorical(le.transform(y_train))
    y_val_oh   = to_categorical(le.transform(y_val))
    y_test_oh  = to_categorical(le.transform(y_test))

    return (X_train, X_val, X_test,
            y_train_oh, y_val_oh, y_test_oh,
            le, scaler)
