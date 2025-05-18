#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import argparse
import sys

def load_rf_model(model_dir='outputs'):
    """RF modelini ve gereken bileşenleri yükler"""
    try:
        # Model dosya yollarını belirle
        rf_path = os.path.join(model_dir, 'rf_model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        # Modeli ve Scaler'ı yükle
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        
        # Sınıf etiketleri (sabit)
        labels = np.array(['GALAXY', 'QSO', 'STAR'])
        
        print(f"RF modeli başarıyla yüklendi: {rf_path}")
        print(f"Model özellikleri:")
        print(f"- Ağaç sayısı: {rf.n_estimators}")
        print(f"- OOB skoru: {rf.oob_score_:.4f}")
        
        return rf, scaler, labels
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {str(e)}")
        return None, None, None

def preprocess_data(df, scaler):
    """Veriyi ön işler ve özellik vektörlerini oluşturur"""
    try:
        print(f"Ön işleme öncesi boyut: {df.shape}")
        
        # Koordinat ve ID sütunlarını kaldır
        cols_to_drop = []
        for col in ['objid', 'specobjid', 'run', 'rerun', 'camcol', 'field', 'ra', 'dec']:
            if col in df.columns:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df = df.drop(cols_to_drop, axis=1)
        
        # Kategorik sütunları ayır
        y = None
        if 'class' in df.columns:
            y = df['class'].copy()
            df = df.drop(['class'], axis=1)
        
        # Renk indekslerini ekle (eğer beş temel filtre varsa)
        if all(band in df.columns for band in ['u', 'g', 'r', 'i', 'z']):
            if 'u_g' not in df.columns:
                df["u_g"] = df["u"] - df["g"]
            if 'g_r' not in df.columns:
                df["g_r"] = df["g"] - df["r"]
            if 'r_i' not in df.columns:
                df["r_i"] = df["r"] - df["i"]
            if 'i_z' not in df.columns:
                df["i_z"] = df["i"] - df["z"]
        
        # Sayısal olmayan veya eksik değerleri kontrol et ve temizle
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
            print(f"Uyarı: Sayısal olmayan sütunlar kaldırılıyor: {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)
        
        # NaN ve sonsuz değerleri kontrol et
        nan_count = df.isna().sum().sum()
        inf_count = ((df == np.inf) | (df == -np.inf)).sum().sum()
        if nan_count > 0 or inf_count > 0:
            print(f"Eksik değerler: {nan_count} NaN, {inf_count} sonsuz")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(df.median(), inplace=True)
        
        # Özellik sütunlarını scaler'da olan sütunlarla eşleştir
        feature_columns = set(df.columns)
        scaler_columns = set(scaler.feature_names_in_)
        
        # Eksik sütunları kontrol et
        missing_columns = scaler_columns - feature_columns
        if missing_columns:
            print(f"Uyarı: Modelin beklediği bazı sütunlar eksik: {missing_columns}")
            # Eksik sütunlar için 0 ile doldur
            for col in missing_columns:
                df[col] = 0
        
        # Fazla sütunları kontrol et
        extra_columns = feature_columns - scaler_columns
        if extra_columns:
            print(f"Uyarı: Modelde olmayan fazla sütunlar kaldırılıyor: {extra_columns}")
            df = df.drop(columns=extra_columns)
        
        # Sütun sıralamasını scaler ile uyumlu hale getir
        df = df[scaler.feature_names_in_]
        
        # Veriyi ölçeklendir
        X = scaler.transform(df)
        print(f"Ön işleme sonrası özellik vektörü boyutu: {X.shape}")
        
        return X, y
        
    except Exception as e:
        print(f"Veri ön işleme sırasında hata: {str(e)}")
        return None, None

def predict_with_rf(X, rf, labels):
    """Random Forest ile tahmin yapar"""
    try:
        # Olasılık değerlerini tahmin et
        rf_probs = rf.predict_proba(X)
        
        # Sınıf tahminleri
        y_pred_indices = rf_probs.argmax(1)
        y_pred_labels = [labels[idx] for idx in y_pred_indices]
        
        return y_pred_labels, y_pred_indices, rf_probs
    except Exception as e:
        print(f"Tahmin sırasında hata: {str(e)}")
        return None, None, None

def evaluate_model(y_true, y_pred, labels=None):
    """Model performansını değerlendirir"""
    if y_true is None or len(y_true) == 0:
        print("Değerlendirme için gerçek etiketler bulunamadı.")
        return
    
    # Doğruluk (accuracy)
    acc = accuracy_score(y_true, y_pred)
    print(f"\nDoğruluk (Accuracy): {acc:.4f} ({acc*100:.2f}%)")
    
    # Detaylı sınıflandırma raporu
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_true, y_pred, target_names=labels if labels is not None else None))
    
    # Karmaşıklık matrisi
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels if labels is not None else 'auto',
                yticklabels=labels if labels is not None else 'auto')
    plt.title('Random Forest Karmaşıklık Matrisi')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.tight_layout()
    plt.show()

def main():
    # Argüman ayrıştırıcı oluştur
    parser = argparse.ArgumentParser(description='Random Forest modeli ile test yapar')
    parser.add_argument('--data', type=str, required=True, help='Test edilecek CSV dosyasının yolu')
    parser.add_argument('--model_dir', type=str, default='outputs', help='Model dosyalarının bulunduğu dizin')
    parser.add_argument('--no_plot', action='store_true', help='Görselleştirmeleri devre dışı bırak')
    
    # Argümanları ayrıştır
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("RANDOM FOREST MODEL TEST ARACI".center(70))
    print("="*70 + "\n")
    
    # Modeli yükle
    rf, scaler, labels = load_rf_model(args.model_dir)
    if rf is None or scaler is None:
        print("Model yüklenemedi. Program sonlandırılıyor.")
        sys.exit(1)
    
    # Veriyi yükle
    try:
        print(f"\nTest verisi yükleniyor: {args.data}")
        df_test = pd.read_csv(args.data)
        print(f"Veri seti boyutu: {df_test.shape[0]} satır, {df_test.shape[1]} sütun")
        
        # İlk birkaç satırı göster
        print("\nVeri örneği:")
        print(df_test.head(3))
        
        # Sınıf dağılımını göster (eğer 'class' sütunu varsa)
        if 'class' in df_test.columns:
            print("\nSınıf dağılımı:")
            class_dist = df_test['class'].value_counts()
            for cls, count in class_dist.items():
                print(f"  {cls}: {count} örnek ({count/len(df_test)*100:.2f}%)")
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {str(e)}")
        sys.exit(1)
    
    # Veriyi ön işle
    X, y_true = preprocess_data(df_test, scaler)
    if X is None:
        print("Veri ön işleme başarısız oldu. Program sonlandırılıyor.")
        sys.exit(1)
    
    # Tahmin yap
    y_pred_labels, y_pred_indices, rf_probs = predict_with_rf(X, rf, labels)
    if y_pred_labels is None:
        print("Tahmin yapılamadı. Program sonlandırılıyor.")
        sys.exit(1)
    
    # Sonuçları göster
    print("\nTahmin sonuçları:")
    if len(y_pred_labels) <= 10:
        for i, pred in enumerate(y_pred_labels):
            print(f"Örnek {i+1}: {pred} (Olasılık: {rf_probs[i][y_pred_indices[i]]:.4f})")
    else:
        print(f"Toplam {len(y_pred_labels)} tahmin yapıldı.")
        class_counts = pd.Series(y_pred_labels).value_counts()
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} örnek ({count/len(y_pred_labels)*100:.2f}%)")
    
    # Değerlendirme
    if y_true is not None:
        # Label encoding testi
        if not isinstance(y_true.iloc[0] if hasattr(y_true, 'iloc') else y_true[0], (int, np.integer)):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(labels)
            y_true = le.transform(y_true)
        
        # Modeli değerlendir
        if not args.no_plot:
            evaluate_model(y_true, y_pred_indices, labels)
        else:
            # Grafik olmadan değerlendirme
            acc = accuracy_score(y_true, y_pred_indices)
            print(f"\nDoğruluk (Accuracy): {acc:.4f} ({acc*100:.2f}%)")
            print("\nSınıflandırma Raporu:")
            print(classification_report(y_true, y_pred_indices, target_names=labels))
    
    print("\nTest işlemi tamamlandı!")

if __name__ == "__main__":
    main()
