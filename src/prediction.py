#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
from astropy import units as u
from PIL import Image
from io import BytesIO
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Özellik vektörü oluşturma - Renk filtreleri ve indeksler
# -------------------------------------------------
def make_feature_vector(u, g, r, i, z, plate=None, mjd=None, fiberid=None, redshift=None):
    """5 temel fotometrik filtreden DataFrame oluşturur
    preprocess_data() fonksiyonu ile tutarlı şekilde çalışacak.
    """
    # Temel 5 fotometrik değer ve ek SDSS parametrelerini içeren DataFrame oluştur
    data = pd.DataFrame({
        'u': [u],
        'g': [g],
        'r': [r],
        'i': [i],
        'z': [z],
        'plate': [plate if plate is not None else 0],
        'mjd': [mjd if mjd is not None else 0],
        'fiberid': [fiberid if fiberid is not None else 0],
        'redshift': [redshift if redshift is not None else 0]
    })
    
    print(f"Temel fotometrik değerler: u={u}, g={g}, r={r}, i={i}, z={z}")
    print(f"Ek SDSS parametreleri: plate={plate}, mjd={mjd}, fiberid={fiberid}, redshift={redshift}")
    print(f"Oluşturulan DataFrame boyutu: {data.shape}")
    
    return data

# ---------------------------------------------------------------------
# Model yükleme işlevi
# ---------------------------------------------------------------------
@st.cache_resource
def load_models(model_dir=None):
    """Eğitilmiş Random Forest modelini yükler"""
    try:
        # Varsayılan model dizini
        if model_dir is None:
            # Şu anki dosyanın bulunduğu dizinden bir üst dizine, oradan da outputs dizinine git
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            model_dir = os.path.join(parent_dir, 'outputs')
        
        # Model dosya yollarını belirle
        rf_path = os.path.join(model_dir, 'rf_model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        
        # Modeli ve Scaler'ı yükle
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        
        # Sınıf etiketleri
        labels = np.array(['GALAXY', 'QSO', 'STAR'])
        
        print(f"Random Forest modeli başarıyla yüklendi: {rf_path}")
        print(f"Scaler başarıyla yüklendi: {scaler_path}")
        
        return rf, scaler, labels
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None, None, None

# ---------------------------------------------------------------------
# Tahmin işlevi
# ---------------------------------------------------------------------
def predict(sample_array, rf, scaler, labels):
    """
    Yeni veri için tahmin yapar
    CSV yükleme bölümü ile aynı tahmin mantığını kullanır
    """
    try:
        # Giriş doğrulama
        if rf is None or scaler is None or labels is None:
            raise ValueError("Model, scaler veya etiketler yüklenemedi")
        
        if sample_array is None or sample_array.size == 0:
            raise ValueError("Geçersiz giriş verisi")
            
        # 1) Ölçeklendirme için özellik sayısı kontrolü
        expected_features = len(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else 13
        actual_features = sample_array.shape[1]
        
        print(f"Özellik kontrolü: Beklenen={expected_features}, Gerçek={actual_features}")
        
        # 2) Özellik sayısını eşitleyelim (gerekirse)
        adjusted_sample = sample_array.copy()
        if actual_features != expected_features:
            print(f"Özellik sayısını ayarlıyorum: {actual_features} -> {expected_features}")
            if actual_features < expected_features:
                # Eksik özellikleri 0 ile doldur
                padding = np.zeros((adjusted_sample.shape[0], expected_features - actual_features))
                adjusted_sample = np.hstack([adjusted_sample, padding])
                print(f"Eksik özellikler 0 ile dolduruldu. Yeni boyut: {adjusted_sample.shape}")
            else:
                # Fazla özellikleri at
                adjusted_sample = adjusted_sample[:, :expected_features]
                print(f"Fazla özellikler atıldı. Yeni boyut: {adjusted_sample.shape}")
        
        # 3) Ölçeklendirme yap (scaler kullan)
        X_scaled = scaler.transform(adjusted_sample)
        print(f"Ölçeklendirilmiş özellik vektörü boyutu: {X_scaled.shape}")
        
        # 4) Tahmin yap (RF modeli ile)
        rf_probs = rf.predict_proba(X_scaled)
        
        # 5) Sonuçları çıkar
        pred_classes_idx = rf_probs.argmax(1)
        pred_class = labels[pred_classes_idx[0]]
        confidence = rf_probs[0, pred_classes_idx[0]]
        
        # 6) Tüm sınıf olasılıklarını hazırla
        class_probs = {label: float(rf_probs[0, i]) for i, label in enumerate(labels)}
        
        print(f"Tahmin: '{pred_class}', Güven: {confidence:.4f}")
        print(f"Tüm sınıf olasılıkları: {class_probs}")
        
        return pred_class, confidence, class_probs
        
    except Exception as e:
        error_msg = f"Tahmin yaparken hata oluştu: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        
        # Hata durumunda varsayılan değer döndür
        dummy_probs = {label: 1.0/len(labels) for label in labels}
        return "HATA", 0.0, dummy_probs

# ---------------------------------------------------------------------
# SDSS Veri Çekme İşlevleri
# ---------------------------------------------------------------------
def get_spectra_link(obj_id):
    """SDSS'ten verilen obj_id için spektrum bağlantısını alır"""
    try:
        return f"https://dr16.sdss.org/optical/spectrum/view/data/format=lite?plateid={obj_id['plate']}&mjd={obj_id['mjd']}&fiberid={obj_id['fiberid']}"
    except Exception as e:
        print(f"Spektrum bağlantısı oluşturulurken hata: {str(e)}")
        return None

def get_sdss_object_by_coords(ra, dec, radius=5.0):
    """SDSS'ten verilen koordinatlar için nesne bilgilerini çeker"""
    try:
        coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
        results = SDSS.query_region(coords, radius=radius*u.arcsec, spectro=True)
        
        if results is None or len(results) == 0:
            return None
        
        # İlk eşleşmeyi al
        return results[0]
    except Exception as e:
        print(f"SDSS veri çekilirken hata: {str(e)}")
        return None

def get_sdss_image(ra, dec, scale=0.3, width=256, height=256):
    """SDSS'ten verilen koordinatlar için gökyüzü görüntüsünü çeker"""
    try:
        image_url = f"http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width={width}&height={height}"
        response = requests.get(image_url)
        
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Görüntü çekilemedi: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Görüntü çekilirken hata: {str(e)}")
        return None

# ---------------------------------------------------------------------
# Veri Görselleştirme İşlevleri
# ---------------------------------------------------------------------
def plot_predictions(pred_class, class_probs):
    """Tahmin sonuçlarını görselleştirir"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    try:
        # Sınıf olasılıkları boş veya None olabilir
        if not class_probs:
            # Boş olasılıklar için varsayılan değerler
            default_labels = ['GALAXY', 'QSO', 'STAR']
            class_probs = {label: 0.0 for label in default_labels}
            class_probs[default_labels[0]] = 1.0  # İlk sınıfa 1.0 olasılık ver
            pred_class = default_labels[0]
        
        # pred_class, class_probs içinde yoksa hata oluşma ihtimali var
        if pred_class not in class_probs:
            # Eğer tahmin edilen sınıf olasılıklarda yoksa, ilk anahtarı kullan
            pred_class = list(class_probs.keys())[0]
            st.warning(f"Tahmin edilen sınıf '{pred_class}', olasılık listesinde bulunamadı. İlk sınıf kullanılıyor.")
        
        # Renk haritası
        colors = {'GALAXY': '#3498db', 'QSO': '#e74c3c', 'STAR': '#2ecc71'}
        bar_colors = [colors.get(cls, '#7f8c8d') for cls in class_probs.keys()]
        
        # Bar plot
        bars = ax.bar(list(class_probs.keys()), list(class_probs.values()), color=bar_colors)
        
        # Tahmin edilen sınıfı vurgula
        idx = list(class_probs.keys()).index(pred_class)
        bars[idx].set_alpha(0.9)
        bars[idx].set_hatch('/')
        
        # Grafik ayarları
        ax.set_title('Sınıf Tahmin Olasılıkları')
        ax.set_ylabel('Olasılık')
        ax.set_ylim(0, 1.0)
        
        # Olasılık değerlerini çubukların üzerine ekle
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    except Exception as e:
        # Hata durumunda bir mesaj göster ama çökmesin
        ax.text(0.5, 0.5, f"Grafik oluşturulamadı: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        # Hata mesajını yazdır
        print(f"plot_predictions fonksiyonunda hata: {str(e)}")
    
    plt.tight_layout()
    return fig

def display_confidence_gauge(confidence):
    """Güven değerini göstermek için ölçek grafiği oluşturur"""
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Ölçek aralığı ve renkler
    cmap = plt.cm.RdYlGn  # Kırmızı-Sarı-Yeşil renk haritası
    norm = plt.Normalize(0, 1)
    
    # Ölçeği çiz
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, norm=norm)
    
    # İşaretçiyi yerleştir
    marker_pos = confidence * fig.get_figwidth() * fig.dpi * 0.8
    marker_pos = min(marker_pos, fig.get_figwidth() * fig.dpi * 0.8)  # Sınırları aşmayı önle
    ax.axvline(marker_pos, color='black', linewidth=3)
    
    # Etiketler
    ax.text(0, 0.5, '0.0', ha='left', va='center', transform=ax.transAxes)
    ax.text(1, 0.5, '1.0', ha='right', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.5, f'{confidence:.2f}', ha='center', va='center', 
            transform=ax.transAxes, fontweight='bold', fontsize=12)
    
    # Eksen gizleme
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------
# Diğer Yardımcı İşlevler
# ---------------------------------------------------------------------
def get_object_info_text(obj_class, confidence):
    """Tahmin edilen nesneyle ilgili açıklayıcı metin oluşturur"""
    info = {
        'GALAXY': "Gökada (Galaksi), yıldızlar, yıldızlararası gaz, toz, karanlık madde ve olası bir süpermasif karadelikten oluşan, kütleçekimi ile bir arada tutulan geniş bir kozmik yapıdır.",
        'QSO': "Quasar (QSO, Yarı-Yıldızsı Nesne), aktif bir gökada çekirdeğidir. Merkezi süpermasif kara deliğe düşen maddenin oluşturduğu ışınımla, evrendeki en parlak nesnelerden biridir.",
        'STAR': "Yıldız, kendi kütleçekimi etkisiyle bir arada tutulan, termonükleer füzyon yoluyla enerji üreten küresel bir gök cismidir.",
        'Bilinmeyen': "Bu gök cisminin türü belirlenemedi veya sınıflandırma sırasında bir hata oluştu."
    }
    
    # Obje sınıfı tanımlı değilse Bilinmeyen olarak göster
    if obj_class not in info:
        obj_class = 'Bilinmeyen'
        
    # Güven değerine göre ek bilgiler
    if confidence <= 0.1:  # Çok düşük güven durumu için özel mesaj
        return "Sınıflandırma yapılamadı veya çok düşük bir güven değeri elde edildi. Lütfen farklı bir veri ile tekrar deneyin."
        
    confidence_info = ""
    if confidence >= 0.95:
        confidence_info = "Bu tahmin çok yüksek bir güvenle yapılmıştır."
    elif confidence >= 0.85:
        confidence_info = "Bu tahmin yüksek bir güvenle yapılmıştır."
    elif confidence >= 0.75:
        confidence_info = "Bu tahmin makul bir güvenle yapılmıştır."
    elif confidence >= 0.6:
        confidence_info = "Bu tahmin orta seviyede bir güvenle yapılmıştır."
    else:
        confidence_info = "Bu tahmin düşük bir güvenle yapılmıştır ve yanlış olabilir."
    
    return f"{info.get(obj_class, 'Bilinmeyen nesne tipi.')} {confidence_info}"
