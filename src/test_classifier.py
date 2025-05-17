#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Tabanlı Astronomik Sınıflandırıcı Test Scripti

Bu script, app.py içerisindeki sınıflandırma işlevlerini bağımsız olarak test etmek için 
kullanılır. Farklı bias düzeltme faktörleri ve model ağırlıkları test edilebilir.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# app.py'deki fonksiyonlara erişim
from app import make_feature_vector, load_models, predict

# Hata yönetimi fonksiyonu (Streamlit olmadan çalışmak için)
def error_print(msg):
    """app.py'deki st.error() fonksiyonunu taklit eder"""
    print(f"HATA: {msg}")

# Modelleri yükle
def load_test_models(model_dir='../outputs'):
    """Eğitilmiş modelleri yükler - Streamlit olmadan çalışacak şekilde"""
    try:
        # Doğru dizin yolunu belirle
        # Göreli yol kullanarak çalışma dizinine göre doğru konumu belirleyelim
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir_abs = os.path.normpath(os.path.join(current_dir, model_dir))
        print(f"Model dizini: {model_dir_abs}")
        
        dnn_path = os.path.join(model_dir_abs, 'dnn_model.keras')
        if not os.path.exists(dnn_path):
            error_print(f"DNN model dosyası bulunamadı: {dnn_path}")
            # Alternatif dosya adlarını kontrol et
            alternative_names = ['dnn_model.h5', 'dnn_model']
            for alt_name in alternative_names:
                alt_path = os.path.join(model_dir_abs, alt_name)
                if os.path.exists(alt_path):
                    error_print(f"Alternatif DNN model dosyası bulundu: {alt_path}")
                    dnn_path = alt_path
                    break
        
        print(f"DNN model yolu: {dnn_path}")
        dnn = load_model(dnn_path, compile=False)  # compile=False parametresi eklendi
        print("DNN modeli başarıyla yüklendi.")
        
        # Random Forest modelini yükle
        rf_path = os.path.join(model_dir_abs, 'rf_model.joblib')
        if not os.path.exists(rf_path):
            error_print(f"RF model dosyası bulunamadı: {rf_path}")
            return None, None, None, None, None
            
        print(f"RF model yolu: {rf_path}")
        rf = joblib.load(rf_path)
        print("RF modeli başarıyla yüklendi.")
          
        # Scaler modelini yükle
        scaler_path = os.path.join(model_dir_abs, 'scaler.joblib')
        if not os.path.exists(scaler_path):
            error_print(f"Scaler dosyası bulunamadı: {scaler_path}")
            return None, None, None, None, None
            
        print(f"Scaler yolu: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print("Scaler başarıyla yüklendi.")

        # Etiketler ve ağırlık
        labels = np.array(['GALAXY', 'QSO', 'STAR'])
        best_w = 0.5  # Varsayılan ağırlık
        
        print("Tüm modeller başarıyla yüklendi.")
        return dnn, rf, scaler, labels, best_w
    except Exception as e:
        error_print(f"Model yüklenirken hata oluştu: {str(e)}")
        import traceback
        error_print(traceback.format_exc())  # Detaylı hata mesajı
        return None, None, None, None, None

# Farklı ağırlık ve bias düzeltme seçenekleriyle tahmin yapan fonksiyon
def predict_with_options(sample_array, dnn, rf, scaler, labels, 
                        dnn_weight=0.5, bias_correction=None, verbose=True):
    """Farklı ağırlık ve bias düzeltme seçenekleriyle tahmin yapar"""
    try:
        """Ölçekle ➜ DNN & RF ➜ Ağırlıklı oy."""
        # 1) StandardScaler
        X = scaler.transform(sample_array)

        # 2) Olasılıklar
        dnn_probs = dnn.predict(X, verbose=0)
        rf_probs = rf.predict_proba(X)
        
        if verbose:
            print(f"DNN tahminleri: {dnn_probs[0]}")
            print(f"RF tahminleri: {rf_probs[0]}")
        
        # 3) Ensemble - belirtilen ağırlık ile
        ens_probs = dnn_weight*dnn_probs + (1-dnn_weight)*rf_probs
        
        # Düzeltme öncesi olasılıkları kaydet
        pre_correction_probs = ens_probs.copy()
        
        # 3.5) Bias düzeltme uygula (eğer belirtilmişse)
        if bias_correction is not None:
            if verbose:
                print(f"Düzeltme öncesi olasılıklar: {ens_probs[0]}")
            
            # Düzeltme uygula
            ens_probs = ens_probs * bias_correction
            
            if verbose:
                print(f"Düzeltme sonrası olasılıklar: {ens_probs[0]}")
        
        # 4) Sonuç
        primary = ens_probs.argmax(1)
        predictions = labels[primary]
        probabilities = ens_probs.max(1) / np.sum(ens_probs, axis=1)  # Normalize edilmiş olasılık
        
        return predictions, probabilities, ens_probs, pre_correction_probs
    except Exception as e:
        error_print(f"Tahmin yaparken hata oluştu: {str(e)}")
        return None, None, None, None

def test_classifier():
    """Sınıflandırıcıyı test eder"""
    # Modelleri yükle
    dnn, rf, scaler, labels, best_w = load_test_models()
    
    if dnn is None or rf is None:
        error_print("Modeller yüklenemedi.")
        return
      # Test verileri - Her bir sınıf için tipik örnekler - SDSS dağılımlarına göre düzenlenmiş
    test_objects = [
        # Tipik bir Galaksi - GALAXY - Daha sönük, galaksiler için tipik renk indeksleriyle
        {"name": "Galaksi", "u": 19.3, "g": 17.6, "r": 16.5, "i": 16.0, "z": 15.7},
        
        # Tipik bir Kuasar - QSO - Karakteristik renk indeksleri ile
        {"name": "Kuasar", "u": 18.4, "g": 18.1, "r": 17.8, "i": 17.5, "z": 17.2},
        
        # Tipik bir Yıldız - STAR - Daha parlak ve yıldızlara özgü renk indeksleriyle
        {"name": "Yıldız", "u": 15.5, "g": 14.4, "r": 13.8, "i": 13.5, "z": 13.4}
    ]
    
    # Farklı bias correction faktörlerini deneyeceğiz
    bias_correction_options = [
        {"name": "Düşük Düzeltme", "factors": np.array([1.2, 1.1, 0.9])},  # Hafif düzeltme
        {"name": "Orta Düzeltme", "factors": np.array([1.5, 1.4, 0.6])},   # Orta düzeltme
        {"name": "Yüksek Düzeltme", "factors": np.array([2.0, 1.8, 0.3])}, # Şiddetli düzeltme
        {"name": "Aşırı Düzeltme", "factors": np.array([3.0, 2.5, 0.1])}   # Aşırı düzeltme
    ]
    
    # DNN ağırlık seçenekleri
    dnn_weight_options = [0.5, 0.3, 0.2, 0.1]
    
    print("\n" + "="*70)
    print("ASTRONOMİK SINIFLANDIRICI TEST BAŞLATILIYOR".center(70))
    print("="*70)
    
    # Her bir test nesnesi için farklı ayarları deneyelim
    for test_obj in test_objects:
        print(f"\n\nTest Nesnesi: {test_obj['name']}")
        print("-" * 40)
        
        # Öznitelikleri oluştur
        features = make_feature_vector(
            test_obj["u"], 
            test_obj["g"], 
            test_obj["r"], 
            test_obj["i"], 
            test_obj["z"]
        )
        
        # 1. Standart tahmin (düzeltme olmadan)
        print("\n1. STANDART TAHMİN (düzeltme olmadan)")
        print("-" * 40)
        
        # Önce olasılıkları manuel olarak hesaplayalım
        X = scaler.transform(features)
        dnn_probs = dnn.predict(X, verbose=0)
        rf_probs = rf.predict_proba(X)
            
        print(f"DNN tahminleri: {dnn_probs[0]}")
        print(f"RF tahminleri: {rf_probs[0]}")
        
        ens_probs = best_w*dnn_probs + (1-best_w)*rf_probs
        print(f"Ensemble olasılıkları (w={best_w}): {ens_probs[0]}")
        print(f"Tahmin: {labels[ens_probs[0].argmax()]} (en yüksek olasılık)")
        
        # 2. Farklı DNN ağırlıkları deneyelim
        print("\n2. FARKLI DNN AĞIRLIKLARI İLE TAHMİN")
        print("-" * 40)
        
        for w in dnn_weight_options:
            ens_probs = w*dnn_probs + (1-w)*rf_probs
            print(f"DNN ağırlığı {w}: {ens_probs[0]} -> {labels[ens_probs[0].argmax()]}")
        
        # 3. Bias düzeltme seçenekleri
        print("\n3. BİAS DÜZELTME SEÇENEKLERİ (DNN ağırlığı = 0.2)")
        print("-" * 40)
        
        # DNN_weight = 0.2 kullan
        base_ens_probs = 0.2*dnn_probs + (1-0.2)*rf_probs
        print(f"Temel ensemble olasılıkları (w=0.2): {base_ens_probs[0]}")
        
        for option in bias_correction_options:
            corrected_probs = base_ens_probs * option["factors"]
            predicted_class = labels[corrected_probs[0].argmax()]
            
            # Olasılıkları normalize et
            norm_probs = corrected_probs[0] / corrected_probs[0].sum()
            
            print(f"\n{option['name']} ({option['factors']}):")
            print(f"  Düzeltilmiş olasılıklar: {corrected_probs[0]}")
            print(f"  Normalize olasılıklar: {norm_probs}")
            print(f"  Tahmin: {predicted_class}")
    
    print("\n" + "="*70)
    print("TEST TAMAMLANDI".center(70))
    print("="*70)

# ---------------------------------------------------------------------
# Kapsamlı test senaryosu
# ---------------------------------------------------------------------
def test_all_scenarios():
    """Tüm test senaryolarını çalıştırır ve sonuçları analiz eder"""
    # Modelleri yükle
    dnn, rf, scaler, labels, _ = load_test_models()
    
    if dnn is None or rf is None:
        error_print("Modeller yüklenemedi. Test durduruluyor.")
        return
    
    # Sonuçları saklayacak veri yapısı
    results = []
    
    # Test edilecek DNN ağırlıkları
    dnn_weights = [0.5, 0.3, 0.2, 0.1]
    
    # Test edilecek bias düzeltme faktörleri
    bias_corrections = [
        None,  # Düzeltme yok
        np.array([1.5, 1.3, 0.6]),  # Hafif düzeltme
        np.array([2.0, 1.8, 0.3]),  # Agresif düzeltme (app.py'de kullanılan)
        np.array([3.0, 2.0, 0.1]),  # Çok agresif düzeltme
    ]    # Test nesneleri - SDSS veri dağılımlarına göre düzenlenmiş
    test_objects = {
        "Galaksi Örneği": {"u": 19.3, "g": 17.6, "r": 16.5, "i": 16.0, "z": 15.7},
        "Kuasar Örneği": {"u": 18.4, "g": 18.1, "r": 17.8, "i": 17.5, "z": 17.2},
        "Yıldız Örneği": {"u": 15.5, "g": 14.4, "r": 13.8, "i": 13.5, "z": 13.4}
    }
    
    # Her nesne için test et
    for obj_name, magnitudes in test_objects.items():
        # Feature vektörü oluştur
        features = make_feature_vector(
            magnitudes['u'], magnitudes['g'], magnitudes['r'], 
            magnitudes['i'], magnitudes['z']
        )
        
        print(f"\n{'='*50}")
        print(f"TEST NESNESI: {obj_name}")
        print(f"{'='*50}")
        
        # Her DNN ağırlığı için test et
        for dnn_w in dnn_weights:
            print(f"\n>> DNN Ağırlığı: {dnn_w}")
            
            # Her bias düzeltme seçeneği için test et
            for i, bias in enumerate(bias_corrections):
                bias_desc = "Yok" if bias is None else f"Senaryo {i}: {bias}"
                print(f"\n> Bias Düzeltme: {bias_desc}")
                
                # Tahmin yap
                predictions, probs, all_probs, pre_corr_probs = predict_with_options(
                    features, dnn, rf, scaler, labels, 
                    dnn_weight=dnn_w, bias_correction=bias,
                    verbose=True
                )
                
                if predictions is not None:
                    print(f"Tahmin: {predictions[0]}, Güven: {probs[0]*100:.2f}%")
                    
                    # Sonuçları kaydet
                    results.append({
                        'Nesne': obj_name,
                        'DNN Ağırlığı': dnn_w,
                        'Bias Düzeltme': str(bias),
                        'Tahmin': predictions[0],
                        'Güven (%)': probs[0]*100,
                        'GALAXY (%)': all_probs[0][0]*100 / np.sum(all_probs[0]),
                        'QSO (%)': all_probs[0][1]*100 / np.sum(all_probs[0]),
                        'STAR (%)': all_probs[0][2]*100 / np.sum(all_probs[0])
                    })
    
    # Sonuçları DataFrame'e dönüştür
    results_df = pd.DataFrame(results)
    
    # Sonuçları göster
    print("\n\n")
    print("TÜM TEST SONUÇLARI:")
    print("="*80)
    print(results_df)
    
    # CSV olarak kaydet
    results_df.to_csv("../outputs/test_results.csv", index=False)
    print("\nSonuçlar 'outputs/test_results.csv' dosyasına kaydedildi.")
    
    # Sonuçların grafiksel analizi
    analyze_test_results(results_df)
    
    return results_df

# ---------------------------------------------------------------------
# Test sonuçlarını analiz et
# ---------------------------------------------------------------------
def analyze_test_results(results_df):
    """Test sonuçlarını grafiklerle analiz eder"""
    try:
        # Sonuç yoksa çık
        if results_df is None or len(results_df) == 0:
            print("Analiz için sonuç bulunamadı.")
            return
        
        print("\nSONUÇLAR ANALİZ EDİLİYOR...")
        
        # 1. Her nesne türü için doğru tahmin yüzdesini hesapla
        expected_class = {
            'Galaksi Örneği': 'GALAXY', 
            'Kuasar Örneği': 'QSO', 
            'Yıldız Örneği': 'STAR'
        }
        
        for obj_name, expected in expected_class.items():
            obj_results = results_df[results_df['Nesne'] == obj_name]
            correct_preds = obj_results[obj_results['Tahmin'] == expected]
            
            if len(obj_results) > 0:
                accuracy = len(correct_preds) / len(obj_results) * 100
                print(f"{obj_name} doğru tahmin oranı: {accuracy:.1f}% ({len(correct_preds)}/{len(obj_results)})")
        
        # 2. En iyi parametreleri bul
        best_configs = {}
        
        for obj_name, expected in expected_class.items():
            obj_results = results_df[results_df['Nesne'] == obj_name]
            correct_preds = obj_results[obj_results['Tahmin'] == expected]
            
            if len(correct_preds) > 0:
                # En yüksek güvenle doğru tahmin
                best_conf = correct_preds.sort_values('Güven (%)', ascending=False).iloc[0]
                
                best_configs[obj_name] = {
                    'DNN Ağırlığı': best_conf['DNN Ağırlığı'],
                    'Bias Düzeltme': best_conf['Bias Düzeltme'],
                    'Güven (%)': best_conf['Güven (%)']
                }
                
                print(f"\n{obj_name} için en iyi konfigürasyon:")
                print(f"  DNN Ağırlığı: {best_conf['DNN Ağırlığı']}")
                print(f"  Bias Düzeltme: {best_conf['Bias Düzeltme']}")
                print(f"  Güven: {best_conf['Güven (%)']: .2f}%")
        
        # 3. Bias düzeltmenin etkisini görselleştir
        plt.figure(figsize=(15, 8))
        
        # Her nesne için ayrı grafik oluştur
        for i, (obj_name, expected) in enumerate(expected_class.items(), 1):
            plt.subplot(1, 3, i)
            
            # Bu nesne için sonuçlar
            obj_results = results_df[results_df['Nesne'] == obj_name]
            
            # Düzeltme faktörlerini kategori olarak ayır
            obj_results['Düzeltme Kategorisi'] = obj_results['Bias Düzeltme'].apply(
                lambda x: 'Yok' if x == 'None' else ('Hafif' if '1.5' in x else ('Orta' if '2.0' in x else 'Ağır'))
            )
            
            # Grafik
            sns.barplot(x='DNN Ağırlığı', y=expected + ' (%)', 
                        hue='Düzeltme Kategorisi', data=obj_results)
            
            plt.title(f"{obj_name} - {expected} Sınıfı Olasılığı")
            plt.ylim(0, 100)
            
            # Grafikte doğru tahminleri işaretle
            correct = obj_results[obj_results['Tahmin'] == expected]
            for _, row in correct.iterrows():
                plt.text(row.name % 4, row[expected + ' (%)'] + 5, '✓', 
                        fontsize=12, ha='center', color='green')
        
        plt.tight_layout()
        plt.savefig('../outputs/bias_correction_effect.png')
        print("\nGrafikler '../outputs/bias_correction_effect.png' olarak kaydedildi.")
        
        # 4. Özet öneriler
        print("\nÖNERİLER:")
        
        # Her sınıf için en iyi konfigürasyonu bul
        for obj_name, config in best_configs.items():
            print(f"- {obj_name} için DNN Ağırlığı = {config['DNN Ağırlığı']} ve {config['Bias Düzeltme']} kullanılabilir")
        
        # Genel öneri
        correct_preds = results_df[results_df['Tahmin'] == results_df['Nesne'].map(lambda x: expected_class[x])]
        if len(correct_preds) > 0:
            # En çok doğru tahmin yapan konfigürasyon grupları
            correct_counts = correct_preds.groupby(['DNN Ağırlığı', 'Bias Düzeltme']).size().reset_index(name='Doğru Sayısı')
            best_overall = correct_counts.sort_values('Doğru Sayısı', ascending=False).iloc[0]
            
            print(f"\nTüm sınıflar için en iyi genel konfigürasyon:")
            print(f"DNN Ağırlığı = {best_overall['DNN Ağırlığı']}, Bias Düzeltme = {best_overall['Bias Düzeltme']}")
            print(f"Bu konfigürasyon {best_overall['Doğru Sayısı']}/{len(results_df)//len(expected_class.keys())} nesne türünü doğru tahmin ediyor.")
    
    except Exception as e:
        error_print(f"Sonuçlar analiz edilirken hata oluştu: {str(e)}")

def test_dynamic_weighting():
    """
    Dinamik ağırlıklandırma stratejisini değerlendiren test fonksiyonu.
    """
    # Modelleri yükle
    dnn, rf, scaler, labels, _ = load_test_models()
    
    # Modelleri kontrol et
    if dnn is None or rf is None:
        error_print("Modeller yüklenemedi. Test durduruluyor.")
        return

    print("\n" + "="*70)
    print("DİNAMİK AĞIRLIKLANDIRMA STRATEJİSİ TESTİ".center(70))
    print("="*70)
    
    # Test nesneleri
    test_objects = [
        # Galaksi örneği - Sönük, yüksek u değeri (>18)
        {"name": "Galaksi", "u": 19.3, "g": 17.6, "r": 16.5, "i": 16.0, "z": 15.7},
        # Kuasar örneği - Orta parlaklık, karakteristik kuasar renkleri
        {"name": "Kuasar", "u": 18.4, "g": 18.1, "r": 17.8, "i": 17.5, "z": 17.2},
        # Yıldız örneği - Parlak (düşük magnitude)
        {"name": "Yıldız", "u": 15.5, "g": 14.4, "r": 13.8, "i": 13.5, "z": 13.4}
    ]
    
    results = []
    
    # Her nesne için test et
    for test_obj in test_objects:
        print(f"\n\nTest Nesnesi: {test_obj['name']}")
        print("-" * 40)
        
        # Özellikleri oluştur
        features = make_feature_vector(
            test_obj["u"], test_obj["g"], test_obj["r"], 
            test_obj["i"], test_obj["z"]
        )
        
        # 1. Farklı sabit DNN ağırlıklarını karşılaştır
        print("\nSabit DNN Ağırlıkları Karşılaştırması:")
        print("-" * 40)
        
        # Standart tahmin
        X = scaler.transform(features)
        dnn_probs = dnn.predict(X, verbose=0)
        rf_probs = rf.predict_proba(X)
        
        print(f"DNN tahminleri: {dnn_probs[0]}")
        print(f"RF tahminleri: {rf_probs[0]}")
        
        # Farklı sabit ağırlıklar dene
        weights = [0.5, 0.3, 0.2, 0.1]
        for w in weights:
            ens_probs = w*dnn_probs + (1-w)*rf_probs
            prediction = labels[ens_probs[0].argmax()]
            print(f"DNN ağırlığı {w}: {prediction}")
        
        # 2. Niye potansiyel STAR olup olmadığını belirle
        is_likely_star = False
        if test_obj["u"] < 16.5 and test_obj["g"] < 15.0 and test_obj["r"] < 14.5:
            is_likely_star = True
        if rf_probs[0][2] > 0.1:
            is_likely_star = True
        if dnn_probs[0][2] > 0.99 and test_obj["u"] < 17.0 and test_obj["r"] < 15.0:
            is_likely_star = True
            
        # 3. Dinamik ağırlık seçimi
        if is_likely_star:
            dnn_weight = 0.5  # Yıldızlar için yüksek DNN ağırlığı
            bias_correction = np.array([0.5, 0.5, 2.0])
        else:
            dnn_weight = 0.3  # Galaksi/Kuasar için düşük DNN ağırlığı
            if rf_probs[0][1] > rf_probs[0][0]:
                bias_correction = np.array([0.8, 1.5, 0.5])
            else:
                bias_correction = np.array([1.5, 0.8, 0.5])
        
        # 4. Dinamik ağırlık ile tahmin yap
        ens_probs = dnn_weight*dnn_probs + (1-dnn_weight)*rf_probs
        print(f"\nDinamik DNN Ağırlığı: {dnn_weight}")
        print(f"Seçilen bias düzeltme: {bias_correction}")
        print(f"Düzeltme öncesi olasılıklar: {ens_probs[0]}")
        
        # 5. Bias düzeltme uygula
        ens_probs = ens_probs * bias_correction
        print(f"Düzeltme sonrası olasılıklar: {ens_probs[0]}")
        
        # 6. Sonuç
        prediction = labels[ens_probs[0].argmax()]
        probability = ens_probs[0].max() / np.sum(ens_probs[0])
        
        print(f"\nNihai tahmin: {prediction} (Güven: {probability*100:.2f}%)")
          # 7. Beklenen sınıf ile karşılaştır
        expected = test_obj["name"].upper()
        if expected == "GALAKSI":
            expected = "GALAXY"
        elif expected == "KUASAR":
            expected = "QSO"
        elif expected == "YILDIZ":
            expected = "STAR"
        
        is_correct = prediction == expected
        print(f"Doğru mu? {'✓' if is_correct else '✗'} (Beklenen: {expected})")
        
        # 8. Sonuçları kaydet
        results.append({
            'Nesne': test_obj["name"],
            'DNN Ağırlığı': dnn_weight,
            'Bias Düzeltme': str(bias_correction),
            'Tahmin': prediction,
            'Beklenen': expected,
            'Doğru': is_correct,
            'Güven': probability*100
        })
    
    # Sonuçları özet olarak göster
    print("\n\n" + "="*70)
    print("SONUÇLAR ÖZET".center(70))
    print("="*70)
    
    correct_count = sum(1 for r in results if r['Doğru'])
    print(f"Doğruluk: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
    
    for r in results:
        print(f"{r['Nesne']}: {r['Tahmin']} (Beklenen: {r['Beklenen']}) - " +
              f"{'✓' if r['Doğru'] else '✗'} - Güven: {r['Güven']:.1f}%")
    
    print("="*70)

    return results

# ---------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
      # Komut satırı argümanlarını tanımla
    parser = argparse.ArgumentParser(description='AI Tabanlı Astronomik Sınıflandırıcı Test Programı')
    parser.add_argument('--test-all', action='store_true', help='Tüm test senaryolarını çalıştır')
    parser.add_argument('--csv', type=str, help='CSV dosyasından test et')
    parser.add_argument('--weight', type=float, default=0.2, help='DNN modeli ağırlığı (CSV testi için)')
    parser.add_argument('--bias', nargs='+', type=float, help='Bias düzeltme faktörleri (GALAXY, QSO, STAR sırasıyla)')
    parser.add_argument('--basic-test', action='store_true', help='Temel testi çalıştır')
    parser.add_argument('--dynamic-test', action='store_true', help='Dinamik ağırlıklandırma testini çalıştır')
    
    args = parser.parse_args()
    
    # Bias düzeltme faktörlerini ayarla
    bias_correction = None
    if args.bias and len(args.bias) == 3:
        bias_correction = np.array(args.bias)
    elif not args.bias:
        # Varsayılan değerler
        bias_correction = np.array([2.0, 1.8, 0.3])
      # Test seçeneklerine göre çalıştır
    if args.test_all:
        print("Tüm test senaryoları çalıştırılıyor...")
        test_all_scenarios()    
    elif args.csv:
        print(f"CSV dosyasından test yapılıyor: {args.csv}")
        print(f"DNN ağırlığı: {args.weight}, Bias düzeltme: {bias_correction}")
        print("CSV ile test fonksiyonu henüz uygulanmadı.")
        # test_with_csv fonksiyonu henüz uygulanmadı, bu kısmı devre dışı bırakıyoruz
    elif args.dynamic_test:
        print("Dinamik ağırlıklandırma testi çalıştırılıyor...")
        test_dynamic_weighting()
    elif args.basic_test:
        print("Temel test çalıştırılıyor...")
        test_classifier()
    else:
        # Argüman verilmemişse, varsayılan olarak dinamik ağırlık testini çalıştır
        print("Hiçbir seçenek belirtilmedi. Dinamik ağırlıklandırma testi çalıştırılıyor...")
        test_dynamic_weighting()


