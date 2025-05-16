import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
from src.app import extract_features_from_photometry, predict, load_models
import warnings
warnings.filterwarnings('ignore')

# Test verileri
examples = {
    "GALAXY": {"u": 19.6, "g": 17.8, "r": 16.9, "i": 16.5, "z": 16.1},
    "QSO": {"u": 17.6, "g": 17.8, "r": 17.9, "i": 17.7, "z": 17.5},
    "STAR": {"u": 16.4, "g": 15.3, "r": 14.9, "i": 14.7, "z": 14.6}
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
    print(f"Çıkarılan özellikler şekli: {features.shape}")
    print(f"İlk 5 özellik: {features[0, :5]}")
    
    # Tahmin yap
    predictions, probabilities, all_probs = predict(features, dnn, rf, labels, best_w)
    
    print(f"Tahmin edilen sınıf: {predictions[0]}")
    print(f"Güven: {probabilities[0]*100:.2f}%")
    
    # Tüm sınıflar için olasılıklar
    for i, label in enumerate(labels):
        print(f"{label}: {all_probs[0][i]*100:.2f}%")
      # Beklenen vs. gerçek
    print(f"Beklenen sınıf: {class_name}, Tahmin edilen sınıf: {predictions[0]}")
    print("--------------------------")

print("\nTest tamamlandı!")

# Özellik mühendisliği analizi - bakımı kolaylaştırmak için mevcut özellikleri gösterelim
print("\n*** ÖZELLİK DETAY ANALİZİ ***")
galaxy_data = pd.DataFrame({
    'petroMag_u': [examples["GALAXY"]["u"]],
    'petroMag_g': [examples["GALAXY"]["g"]],
    'petroMag_r': [examples["GALAXY"]["r"]],
    'petroMag_i': [examples["GALAXY"]["i"]],
    'petroMag_z': [examples["GALAXY"]["z"]]
})

features = extract_features_from_photometry(galaxy_data)
print(f"Toplam özellik sayısı: {features.shape[1]}")

# Özellik adlarını oluşturalım
basic_features = ["u", "g", "r", "i", "z"]
color_features = ["u-g", "g-r", "r-i", "i-z"]
ratio_features = ["u/g", "g/r", "r/i", "i/z"]
poly_features = ["(u-g)²", "(g-r)²"]

all_feature_names = basic_features + color_features + ratio_features + poly_features

# Modelin beklediği özellik sayısını kontrol edelim
if features.shape[1] != len(all_feature_names):
    print(f"UYARI: Özellik sayısı eşleşmiyor! Çıkarılan: {features.shape[1]}, Beklenen: {len(all_feature_names)}")
    print("Eksik veya fazla özellikler var.")
else:
    print("Özellik sayısı beklenen değerde.")

# Özellikleri gösterelim
for i, name in enumerate(all_feature_names):
    if i < features.shape[1]:
        print(f"{i+1:2d}. {name:6s}: {features[0, i]:.6f}")
    else:
        print(f"{i+1:2d}. {name:6s}: EKSIK")

# DNN modelindeki katmanları listele
print("\nDNN Model Mimarisi:")
for i, layer in enumerate(dnn.layers):
    print(f"Katman {i+1}: {layer.name}, Tür: {layer.__class__.__name__}")

# Model analizi: Son çıkış katmanının ağırlıklarını inceleme
print("\n*** MODEL ÇIKIŞ KATMANI ANALİZİ ***")
output_layer = dnn.layers[-1]
weights = output_layer.get_weights()
if len(weights) > 0:
    W = weights[0]  # Ağırlıklar
    b = weights[1]  # Bias değerleri
    
    print(f"Çıkış katmanı boyutu: {W.shape}")
    print(f"Bias değerleri (sınıf eğilimleri): {b}")
    
    # Class bias analizi
    class_names = {0: "GALAXY", 1: "QSO", 2: "STAR"}
    print("\nSınıf eğilimleri (bias değerleri):")
    for i, bias in enumerate(b):
        print(f"{class_names[i]}: {bias:.4f}")
    
    print("\nBias farkları (pozitif değer eğilim gösterir):")
    print(f"GALAXY - QSO: {b[0] - b[1]:.4f}")
    print(f"GALAXY - STAR: {b[0] - b[2]:.4f}")
    print(f"QSO - STAR: {b[1] - b[2]:.4f}")

# Random Forest model analizi
print("\n*** RANDOM FOREST MODEL ANALİZİ ***")
if hasattr(rf, "feature_importances_"):
    importances = rf.feature_importances_
    
    # Özellik önem sıralaması
    feature_names = basic_features + color_features + ratio_features + poly_features
    feature_importance = [(name, imp) for name, imp in zip(feature_names, importances)]
    sorted_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    print("Özellik önem sıralaması (ilk 5):")
    for name, imp in sorted_importance[:5]:
        print(f"{name}: {imp:.4f}")
    
    # Sınıf olasılıkları dağılımı
    print("\nSınıf olasılıkları dağılımı:")
    # Örnek verileri için RF model çıktıları
    for class_name, test_data in examples.items():
        phot_data = pd.DataFrame({
            'petroMag_u': [test_data['u']],
            'petroMag_g': [test_data['g']],
            'petroMag_r': [test_data['r']],
            'petroMag_i': [test_data['i']],
            'petroMag_z': [test_data['z']]
        })
        
        # Özellikleri çıkar
        features = extract_features_from_photometry(phot_data)
        
        # RF modeli ile tahmin
        rf_probs = rf.predict_proba(features)[0]
        
        print(f"{class_name} örneği RF tahminleri:")
        for i, prob in enumerate(rf_probs):
            print(f"  {labels[i]}: {prob*100:.2f}%")
