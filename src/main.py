import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from prepare_data import load_and_prepare
from model import build_model


def main():
    # ---------------------------------------------------------------------
    # 0) Dosya yolları ve klasör hazırlığı
    # ---------------------------------------------------------------------
    data_path = 'data/skyserver.csv'  # proje kökünden göreceli
    out_dir   = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) Veri yükle & ölçekle & one‑hot etiketle
    # ---------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, df_full = \
        load_and_prepare(data_path)

    y_train_lbl = y_train.argmax(1)
    y_val_lbl   = y_val.argmax(1)
    y_test_lbl  = y_test.argmax(1)

    # ---------------------------------------------------------------------
    # 2) Derin Sinir Ağı
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # 3) Random Forest (CPU) eğitimi
    # ---------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        oob_score=True
    )
    rf.fit(X_train, y_train_lbl)
    print(f"RF OOB accuracy: {rf.oob_score_:.4f}")

    # ---------------------------------------------------------------------
    # 4) Test kümesinde DNN, RF ve Ensemble karşılaştırması
    # ---------------------------------------------------------------------
    dnn_probs = dnn.predict(X_test)          # (n, 3)
    rf_probs  = rf.predict_proba(X_test)     # (n, 3)

    # Eşit ağırlıklı yumuşak oylama
    ens_probs = (dnn_probs + rf_probs) / 2.0
    dnn_pred  = dnn_probs.argmax(1)
    rf_pred   = rf_probs.argmax(1)
    ens_pred  = ens_probs.argmax(1)

    acc_dnn = (dnn_pred == y_test_lbl).mean()*100
    acc_rf  = (rf_pred  == y_test_lbl).mean()*100
    acc_ens = (ens_pred == y_test_lbl).mean()*100

    print(f"DNN  Test Accuracy : {acc_dnn:6.3f}%")
    print(f"RF   Test Accuracy : {acc_rf :6.3f}%")
    print(f"ENS  Test Accuracy : {acc_ens:6.3f}%  <-- En iyi")

    # ---------------------------------------------------------------------
    # 5) Confusion matrix (ensemble)
    # ---------------------------------------------------------------------
    labels = np.unique(df_full['class'])
    cm = confusion_matrix(y_test_lbl, ens_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix — Ensemble')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_ens.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------------------
    # 6) Eğitim doğruluk eğrileri (DNN)
    # ---------------------------------------------------------------------
    hist_len = len(history.history['categorical_accuracy'])
    epochs_arr = range(1, hist_len + 1)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(epochs_arr, history.history['categorical_accuracy'], 'r-', label='Training')
    ax.plot(epochs_arr, history.history['val_categorical_accuracy'], 'b-', label='Validation')
    ax.set_xlabel('Epochs'); ax.set_ylabel('Accuracy'); ax.set_title('DNN Accuracy')
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{out_dir}/accuracy_dnn.png", dpi=150)
    plt.show()

    # ---------------------------------------------------------------------
    # 7) Modelleri kaydet
    # ---------------------------------------------------------------------
    dnn.save(f"{out_dir}/dnn_model.keras")
    joblib.dump(rf, f"{out_dir}/rf_model.joblib")
    print("[+] Modeller kaydedildi → outputs/")


if __name__ == '__main__':
    main()
