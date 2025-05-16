import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

from prepare_data import load_and_prepare
from model import build_model

# ------------------------------------------------------------------
# Helper will be defined after models are trained
# ------------------------------------------------------------------

def main():
    # GPU kullanımını optimize et
    print("TensorFlow sürümü:", tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # GPU bellek büyümesini ayarla
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU kullanıma hazır: {len(gpus)} adet GPU bulundu")
        except RuntimeError as e:
            print(f"GPU ayarlanırken hata: {e}")
    else:
        print("GPU bulunamadı, CPU kullanılacak.")
    
    # ------------------------------------------------------------------
    # 0) Paths
    # ------------------------------------------------------------------
    data_path = 'data/skyserver.csv'
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)    # ------------------------------------------------------------------
    # 1) LOAD GALAXY/QSO/STAR DATA
    # ------------------------------------------------------------------
    X_tr, X_val, X_te, y_tr, y_val, y_te, df_full, scaler = load_and_prepare(data_path)
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
    # 6) SAVE MODELS
    # ------------------------------------------------------------------
    dnn.save(f"{out_dir}/dnn_model.keras")
    import joblib
    joblib.dump(rf, f"{out_dir}/rf_model.joblib")
    joblib.dump(scaler, f"{out_dir}/scaler.joblib")
    
    # İşlem tamamlandı mesajı
    print("\nTemel sınıflandırma modeli eğitimi tamamlandı!")
    print(f"Modeller {out_dir} klasörüne kaydedildi.")
    
    # ------------------ helper for UI / further use -----------------
    global full_predict
    def full_predict(sample_array):
        """Return GALAXY/QSO/STAR classification for each input row."""
        p_ens = best_w*dnn.predict(sample_array) + (1-best_w)*rf.predict_proba(sample_array)
        primary = p_ens.argmax(1)
        out = [labels[cls] for cls in primary]
        return out

    # leave models in global scope for interactive sessions
    globals().update({'dnn': dnn, 'rf': rf, 'best_w': best_w,
                      'labels': labels})
    
    print("\n" + "="*70)
    print("MODEL EĞİTİMİ TAMAMLANDI".center(70))
    print("="*70)
    print(f"\nTüm modeller '{out_dir}' klasörüne kaydedildi.")
    print("\nÖnemli dosyalar:")
    print(f"- {out_dir}/dnn_model.keras: Ana DNN modeli")
    print(f"- {out_dir}/rf_model.joblib: Random Forest modeli")
    
    # Modelinizi kullanmak için:
    print("\nBu modeli kullanmak için:")
    print(">>> from main import full_predict")
    print(">>> sonuclar = full_predict(yeni_veriler)")

def print_helper():
    """Kaydedilen modeller hakkında bilgi verir"""
    out_dir = 'outputs'
    print("\nKaydedilen dosyalar:")
    print(f"- {out_dir}/dnn_model.keras: Derin sinir ağı modeli")
    print(f"- {out_dir}/rf_model.joblib: Random Forest modeli")
    print(f"- {out_dir}/[çeşitli görseller].png: Performans grafikleri")

if __name__ == '__main__':
    # Gerekli kütüphaneleri kontrol et
    try:
        print("\n--- Bağımlılıklar kontrol ediliyor ---")
        # TensorFlow zaten kontrol edildi
        
        # scikit-learn
        import sklearn
        print(f"scikit-learn sürümü: {sklearn.__version__}")
        
        # Gerekli diğer kütüphaneler
        import matplotlib
        print(f"matplotlib sürümü: {matplotlib.__version__}")
        
        import pandas
        print(f"pandas sürümü: {pandas.__version__}")
        
        # Joblib
        import joblib
        print(f"joblib sürümü: {joblib.__version__}")
        
        print("Tüm gerekli kütüphaneler mevcut.\n")
    except ImportError as e:
        print(f"Eksik kütüphane bulundu: {e}")
        print("pip install scikit-learn pandas matplotlib joblib tensorflow seaborn")
    
    # Ana modellerin eğitimi
    print("\n" + "="*70)
    print("ASTRONOMİK SINIFLANDIRICI EĞİTİMİ BAŞLATILIYOR".center(70))
    print("="*70)
      
    # Ana modeli çalıştır
    main()
    
    print("\nİşlemler tamamlandı! Sonuçlar 'outputs' klasöründe.")
