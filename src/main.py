#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import platform
import sys
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
import sklearn
from datetime import datetime

from prepare_data import load_and_prepare

# ------------------------------------------------------------------
# Helper will be defined after the model is trained
# ------------------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("GÖKSEL NESNE SINIFLANDIRICI - SADECE RANDOM FOREST".center(70))
    print("="*70 + "\n")
    
    # ------------------------------------------------------------------
    # 0) Paths
    # ------------------------------------------------------------------
    data_path = 'data/skyserver.csv'
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    
    # ------------------------------------------------------------------
    # 1) LOAD GALAXY/QSO/STAR DATA
    # ------------------------------------------------------------------
    print("Veri yükleniyor ve hazırlanıyor...")
    X_tr, X_val, X_te, y_tr, y_val, y_te, df_full, scaler = load_and_prepare(data_path)
    y_tr_lbl = y_tr.argmax(1)
    y_val_lbl = y_val.argmax(1)
    y_te_lbl = y_te.argmax(1)

    # ------------------------------------------------------------------
    # 2) RANDOM FOREST
    # ------------------------------------------------------------------
    print("\nRandom Forest modeli eğitiliyor...")
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        oob_score=True)
    
    start_time = time.time()
    rf.fit(X_tr, y_tr_lbl)
    train_time = time.time() - start_time
    
    print(f"RF OOB accuracy: {rf.oob_score_:.4f}")
    print(f"Eğitim süresi: {train_time:.2f} saniye")

    # ------------------------------------------------------------------
    # 3) VALİDASYON SETİ PERFORMANSI
    # ------------------------------------------------------------------
    print("\nValidasyon seti üzerinde performans değerlendiriliyor...")
    rf_val = rf.predict_proba(X_val)
    val_acc = (rf_val.argmax(1) == y_val_lbl).mean() * 100
    print(f"Validasyon doğruluğu: {val_acc:.2f}%")

    # ------------------------------------------------------------------
    # 4) TEST SETİ PERFORMANSI
    # ------------------------------------------------------------------
    print("\nTest seti üzerinde performans değerlendiriliyor...")
    rf_probs = rf.predict_proba(X_te)
    rf_acc = (rf_probs.argmax(1) == y_te_lbl).mean() * 100
    print(f"Test doğruluğu: {rf_acc:.2f}%")

    # Sınıf etiketleri
    labels = np.unique(df_full['class'])
    
    # Karmaşıklık matrisi
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_te_lbl, rf_probs.argmax(1)), annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.title('Random Forest Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/rf_confusion_matrix.png", dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # 5) MODEL KAYDETME
    # ------------------------------------------------------------------
    try:
        print("\nModel kaydediliyor...")
        import joblib
        
        # Modeli kaydet
        joblib.dump(rf, f"{out_dir}/rf_model.joblib")
        print(f"Random Forest modeli başarıyla kaydedildi: {out_dir}/rf_model.joblib")
        
        # Scaler'ı kaydet - verinin ölçeklendirilmesi için gerekli
        joblib.dump(scaler, f"{out_dir}/scaler.joblib")
        print(f"Scaler başarıyla kaydedildi: {out_dir}/scaler.joblib")
    
    except Exception as e:
        print(f"\nModel kaydedilirken hata oluştu: {e}")
    
    # İşlem tamamlandı mesajı
    print("\nRandom Forest modeli eğitimi tamamlandı!")
    print(f"Model {out_dir} klasörüne kaydedildi.")
    
    # ------------------ helper for UI / further use -----------------
    global full_predict
    def full_predict(sample_array):
        """Return GALAXY/QSO/STAR classification for each input row."""
        # Veriyi ölçeklendir
        X_scaled = scaler.transform(sample_array)
        
        # RF tahminleri
        probs = rf.predict_proba(X_scaled)
        predictions = probs.argmax(1)
        
        # Etiketlere çevir
        out = [labels[cls] for cls in predictions]
        return out, probs

    # leave model in global scope for interactive sessions
    globals().update({'rf': rf, 'labels': labels, 'scaler': scaler})
    
    print("\n" + "="*70)
    print("MODEL EĞİTİMİ TAMAMLANDI".center(70))
    print("="*70)
    print(f"\nRandom Forest modeli '{out_dir}' klasörüne kaydedildi.")
    print("\nÖnemli dosyalar:")
    print(f"- {out_dir}/rf_model.joblib: Random Forest modeli")
    print(f"- {out_dir}/scaler.joblib: Özellik ölçekleyici")
    print(f"- {out_dir}/rf_confusion_matrix.png: Karmaşıklık matrisi")
    
    # Modeli kullanmak için:
    print("\nBu modeli kullanmak için:")
    print(">>> from main import full_predict")
    print(">>> tahminler, olasılıklar = full_predict(yeni_veriler)")



def print_helper():
    """Kaydedilen model hakkında bilgi verir"""
    out_dir = 'outputs'
    print("\nKaydedilen dosyalar:")
    print(f"- {out_dir}/rf_model.joblib: Random Forest modeli")
    print(f"- {out_dir}/scaler.joblib: Özellik ölçekleyici")
    print(f"- {out_dir}/rf_confusion_matrix.png: Karmaşıklık matrisi")

if __name__ == "__main__":
    main()
