# main.py — DNN + Random Forest with validation‑weighted ensemble

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from prepare_data import load_and_prepare
from model import build_model
from prepare_data import load_and_prepare, load_star_subset
from star_model import build_star_model


def main():
    # ------------------------------------------------------------------
    # 0) Yol ve klasör
    # ------------------------------------------------------------------
    data_path = 'data/skyserver.csv'
    data_path_star = 'data/star_subtypes.csv'
    out_dir   = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Veri hazırla
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, df_full = \
        load_and_prepare(data_path)

    # Etiketleri argmax formatına çevir
    y_train_lbl = y_train.argmax(1)
    y_val_lbl   = y_val.argmax(1)
    y_test_lbl  = y_test.argmax(1)

    # ------------------------------------------------------------------
    # 2) DNN modeli
    # ------------------------------------------------------------------
    dnn = build_model(X_train.shape[1], y_train.shape[1])
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]
    history = dnn.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # ------------------------------------------------------------------
    # 3) Random Forest modeli
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        oob_score=True
    )
    rf.fit(X_train, y_train_lbl)
    print(f"RF OOB accuracy: {rf.oob_score_:.4f}")

    # ------------------------------------------------------------------
    # 4) Validation set üzerinde en iyi ağırlığı bul
    # ------------------------------------------------------------------
    dnn_val_probs = dnn.predict(X_val)
    rf_val_probs  = rf.predict_proba(X_val)

    best_w, best_acc = 0.5, 0.0
    for w in np.linspace(0.1, 0.9, 9):
        blended = w * dnn_val_probs + (1 - w) * rf_val_probs
        acc = (blended.argmax(1) == y_val_lbl).mean()
        if acc > best_acc:
            best_w, best_acc = w, acc

    print(f"[Val] Best DNN weight: {best_w:.2f}  (val acc={best_acc*100:.2f}%)")

    # ------------------------------------------------------------------
    # 5) Test kümesinde DNN, RF ve Ensemble karşılaştırması
    # ------------------------------------------------------------------
    dnn_probs = dnn.predict(X_test)
    rf_probs  = rf.predict_proba(X_test)

    dnn_pred  = dnn_probs.argmax(1)
    rf_pred   = rf_probs.argmax(1)

    # Val‑tabanlı ağırlık
    ens_probs = best_w * dnn_probs + (1 - best_w) * rf_probs
    ens_pred  = ens_probs.argmax(1)

    acc_dnn = (dnn_pred == y_test_lbl).mean() * 100
    acc_rf  = (rf_pred  == y_test_lbl).mean() * 100
    acc_ens = (ens_pred == y_test_lbl).mean() * 100

    print(f"DNN  Test Accuracy : {acc_dnn:6.3f}%")
    print(f"RF   Test Accuracy : {acc_rf :6.3f}%")
    print(f"BEST‑W ENS Accuracy : {acc_ens:6.3f}%  <-- rapor")

    # ------------------------------------------------------------------
    # 6) Confusion matrix (ensemble)
    # ------------------------------------------------------------------
    labels = np.unique(df_full['class'])
    cm = confusion_matrix(y_test_lbl, ens_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix — Best‑W Ensemble')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_ens_bestw.png", dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # 7) DNN eğitim eğrisi
    # ------------------------------------------------------------------
    epochs_arr = range(1, len(history.history['categorical_accuracy']) + 1)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(epochs_arr, history.history['categorical_accuracy'], 'r-', label='Training')
    ax.plot(epochs_arr, history.history['val_categorical_accuracy'], 'b-', label='Validation')
    ax.set_xlabel('Epochs'); ax.set_ylabel('Accuracy'); ax.set_title('DNN Accuracy')
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{out_dir}/accuracy_dnn.png", dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # 8) Modelleri kaydet
    # ------------------------------------------------------------------
    dnn.save(f"{out_dir}/dnn_model.keras")
    joblib.dump(rf, f"{out_dir}/rf_model.joblib")
    print("[+] Models are saved → outputs/")

    # ------------------ STAR SUBCLASS MODEL --------------------------
    X_s_tr, X_s_val, X_s_te, y_s_tr, y_s_val, y_s_te, le_star, scaler_star = \
        load_star_subset(data_path_star)

    star_net = build_star_model(X_s_tr.shape[1], y_s_tr.shape[1])
    hist_star = star_net.fit(
        X_s_tr, y_s_tr,
        epochs=30,
        batch_size=64,
        validation_data=(X_s_val, y_s_val),
        verbose=1
    )
    star_acc = (star_net.predict(X_s_te).argmax(1) ==
                y_s_te.argmax(1)).mean()*100
    print(f"STAR subtype Test Acc: {star_acc:.2f}%")

    # Kaydet
    star_net.save(f"{out_dir}/star_model.keras")
    joblib.dump(le_star,  f"{out_dir}/star_label_enc.joblib")
    joblib.dump(scaler_star, f"{out_dir}/star_scaler.joblib")

    def full_predict(sample_array):
    """
    sample_array : shape (n_samples, feature_dim)  scaled with the SAME scaler
    returns primary label or star subtype
    """
    # 1) Ensemble ile galaxy/qso/star
    p_dnn = dnn.predict(sample_array)
    p_rf  = rf.predict_proba(sample_array)
    p_ens = best_w * p_dnn + (1 - best_w) * p_rf
    primary = p_ens.argmax(1)

    STAR_ID = np.where(labels == "STAR")[0][0]   # genelde 2
    output = []
    for i, cls in enumerate(primary):
        if cls == STAR_ID:
            x_star = scaler_star.transform(sample_array[i:i+1])
            sub_id = star_net.predict(x_star).argmax(1)[0]
            sub_lbl = le_star.inverse_transform([sub_id])[0]
            output.append(f"STAR-{sub_lbl}")
        else:
            output.append(labels[cls])
    return output

if __name__ == '__main__':
    main()
