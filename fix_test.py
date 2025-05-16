import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from src.app import extract_features_from_photometry, predict, load_models

# Uyarıları kapat
import warnings
warnings.filterwarnings('ignore')

# Bias düzeltme faktörlerini tanımla (GALAXY için düşürücü, diğerleri için artırıcı)
bias_correction_factors = {
    "Default": np.array([0.7, 1.2, 1.3]),  # Uygulama için kullanılan faktörler
    "Strong": np.array([0.5, 1.4, 1.5]),   # Daha agresif düzeltme
    "Mild": np.array([0.8, 1.1, 1.1]),     # Daha hafif düzeltme
    "None": np.array([1.0, 1.0, 1.0])      # Düzeltme yok
}

# Test verileri - astronomik olarak anlamlı örnekler
examples = {
    "GALAXY": {"u": 19.6, "g": 17.8, "r": 16.9, "i": 16.5, "z": 16.1},  # Tipik bir galaksi örneği
    "QSO": {"u": 17.6, "g": 17.8, "r": 17.9, "i": 17.7, "z": 17.5},     # Tipik bir quasar örneği (u-g değeri düşük)
    "STAR": {"u": 16.4, "g": 15.3, "r": 14.9, "i": 14.7, "z": 14.6}     # Tipik bir yıldız örneği
}

# Dosya yolları
model_dir = 'outputs'
dnn_path = os.path.join(model_dir, 'dnn_model.keras')
rf_path = os.path.join(model_dir, 'rf_model.joblib')

# Modelleri yükle
print("Modelleri yüklüyorum...")
dnn, rf, labels, best_w = load_models()

if dnn is None or rf is None:
    print("Modeller yüklenemedi!")
    exit(1)

print(f"Modeller başarıyla yüklendi.")
print(f"Etiketler: {labels}")
print(f"En iyi ağırlık: {best_w}")

# Model özellikleri
print(f"\nDNN giriş boyutu: {dnn.input_shape}")
print(f"DNN çıkış boyutu: {dnn.output_shape}")

# Farklı bias düzeltme seviyeleriyle test et
for correction_name, correction_factor in bias_correction_factors.items():
    print(f"\n--- {correction_name} Bias Düzeltme İle Test ({correction_factor}) ---")
    
    # Her sınıf için test
    for class_name, test_data in examples.items():
        print(f"\n----- {class_name} Sınıfı Testi -----")
        # Örnek DataFrame
        phot_data = pd.DataFrame({
            'petroMag_u': [test_data['u']],
            'petroMag_g': [test_data['g']],
            'petroMag_r': [test_data['r']],
            'petroMag_i': [test_data['i']],
            'petroMag_z': [test_data['z']]
        })
        
        # Özellikleri çıkar
        features = extract_features_from_photometry(phot_data)
        
        # Manuel düzeltme yaparak tahmin yap
        # DNN ve RF tahminlerini al
        dnn_probs = dnn.predict(features, verbose=0)
        rf_probs = rf.predict_proba(features)
        
        # Ensemble tahminini hesapla
        ens_probs = best_w * dnn_probs + (1 - best_w) * rf_probs
        
        # Düzeltilmiş olasılıkları hesapla ve normalize et
        corrected_probs = ens_probs * correction_factor.reshape(1, -1)
        corrected_probs = corrected_probs / corrected_probs.sum(axis=1, keepdims=True)
        
        # Tahmin edilen sınıfı bul
        primary = corrected_probs.argmax(1)
        predictions = [labels[cls] for cls in primary]
        probabilities = corrected_probs.max(axis=1)
        
        # Sonuçları göster
        print(f"Tahmin edilen sınıf: {predictions[0]}")
        print(f"Güven: {probabilities[0]*100:.2f}%")
        
        # Tüm sınıflar için olasılıklar
        print("Sınıf olasılıkları:")
        for i, label in enumerate(labels):
            print(f"{label}: {corrected_probs[0][i]*100:.2f}%")
        
        # Beklenen vs. gerçek
        print(f"Beklenen sınıf: {class_name}, Tahmin edilen sınıf: {predictions[0]}")
        
        if class_name == predictions[0]:
            print("✓ DOĞRU TAHMIN!")
        else:
            print("✗ YANLIŞ TAHMIN!")
        
        print("--------------------------")

print("\nTest tamamlandı!")
