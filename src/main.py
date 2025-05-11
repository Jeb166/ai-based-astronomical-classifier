# main.py — DNN + RF ensemble + STAR subtype model (fixed indent)

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from prepare_data import load_and_prepare, load_star_subset
from model import build_model
from star_model import build_star_model

# ------------------------------------------------------------------
# Helper will be defined after models are trained
# ------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------
    # 0) Paths
    # ------------------------------------------------------------------
    data_path       = 'data/skyserver.csv'
    data_path_star  = 'data/star_subtypes.csv'
    out_dir         = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) LOAD GAL/QSO/STAR DATA
    # ------------------------------------------------------------------
    X_tr, X_val, X_te, y_tr, y_val, y_te, df_full = load_and_prepare(data_path)
    y_tr_lbl  = y_tr.argmax(1)
    y_val_lbl = y_val.argmax(1)
    y_te_lbl  = y_te.argmax(1)

    # ------------------------------------------------------------------
    # 2) DNN
    # ------------------------------------------------------------------
    dnn = build_model(X_tr.shape[1], y_tr.shape[1])
    dnn.fit(
        X_tr, y_tr,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                   ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)],
        verbose=1
    )

    # ------------------------------------------------------------------
    # 3) RANDOM FOREST
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        oob_score=True)
    rf.fit(X_tr, y_tr_lbl)
    print(f"RF OOB accuracy: {rf.oob_score_:.4f}")

    # ------------------------------------------------------------------
    # 4) VALIDATION‑BASED BEST WEIGHT
    # ------------------------------------------------------------------
    dnn_val = dnn.predict(X_val)
    rf_val  = rf.predict_proba(X_val)
    best_w, best_acc = 0.5, 0.0
    for w in np.linspace(0.1, 0.9, 9):
        if (proba_acc := ((w*dnn_val+(1-w)*rf_val).argmax(1)==y_val_lbl).mean()) > best_acc:
            best_w, best_acc = w, proba_acc
    print(f"[Val] Best DNN weight: {best_w:.2f}  (val acc={best_acc*100:.2f}%)")

    # ------------------------------------------------------------------
    # 5) TEST SET SCORES
    # ------------------------------------------------------------------
    dnn_probs = dnn.predict(X_te)
    rf_probs  = rf.predict_proba(X_te)
    ens_probs = best_w*dnn_probs + (1-best_w)*rf_probs
    dnn_acc = (dnn_probs.argmax(1)==y_te_lbl).mean()*100
    rf_acc  = (rf_probs.argmax(1)==y_te_lbl).mean()*100
    ens_acc = (ens_probs.argmax(1)==y_te_lbl).mean()*100
    print(f"DNN  Test Accuracy : {dnn_acc:6.3f}%")
    print(f"RF   Test Accuracy : {rf_acc :6.3f}%")
    print(f"BEST‑W ENS Accuracy : {ens_acc:6.3f}%")

    labels = np.unique(df_full['class'])
    sns.heatmap(confusion_matrix(y_te_lbl, ens_probs.argmax(1)), annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix — Best‑W Ensemble')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_ens_bestw.png", dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # 6) SAVE GAL/QSO/STAR MODELS
    # ------------------------------------------------------------------
    dnn.save(f"{out_dir}/dnn_model.keras")
    joblib.dump(rf, f"{out_dir}/rf_model.joblib")

    # ------------------------------------------------------------------
    # 7) STAR SUB‑CLASS MODEL
    # ------------------------------------------------------------------
    Xs_tr, Xs_val, Xs_te, ys_tr, ys_val, ys_te, le_star, scaler_star = load_star_subset(data_path_star)
    star_net = build_star_model(Xs_tr.shape[1], ys_tr.shape[1])
    y_int = ys_tr.argmax(1)
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_int), y=y_int)
    cw_dict = dict(enumerate(cw))
    star_net.fit(
        Xs_tr, ys_tr,
        epochs=100,  # Daha fazla epoch
        batch_size=32,  # Daha küçük batch size
        validation_data=(Xs_val, ys_val),
        class_weight=cw_dict,
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
        ],
        verbose=1
    )
    star_acc = (star_net.predict(Xs_te).argmax(1)==ys_te.argmax(1)).mean()*100
    print(f"STAR subtype Test Acc: {star_acc:.2f}%")
    star_net.save(f"{out_dir}/star_model.keras")
    joblib.dump(le_star, f"{out_dir}/star_label_enc.joblib")
    joblib.dump(scaler_star, f"{out_dir}/star_scaler.joblib")

    # ------------------ helper for UI / further use -----------------
    global full_predict
    def full_predict(sample_array):
        """Return GALAXY/QSO/STAR‑subtype for each input row (2‑stage)."""
        p_ens = best_w*dnn.predict(sample_array) + (1-best_w)*rf.predict_proba(sample_array)
        primary = p_ens.argmax(1)
        STAR_ID = np.where(labels=='STAR')[0][0]
        out = []
        for i, cls in enumerate(primary):
            if cls == STAR_ID:
                x_s = scaler_star.transform(sample_array[i:i+1])
                sub_id = star_net.predict(x_s).argmax(1)[0]
                out.append(f"STAR-{le_star.inverse_transform([sub_id])[0]}")
            else:
                out.append(labels[cls])
        return out

    # leave models in global scope for interactive sessions
    globals().update({'dnn': dnn, 'rf': rf, 'best_w': best_w,
                      'labels': labels, 'star_net': star_net,
                      'scaler_star': scaler_star, 'le_star': le_star})

if __name__ == '__main__':
    main()
