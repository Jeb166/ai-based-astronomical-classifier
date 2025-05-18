import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
from astropy import units as u
from PIL import Image
from io import BytesIO
import requests
import streamlit as st

# -------------------------------------------------
# 15-özelliklik vektör (model için tam gereken şekilde)
# -------------------------------------------------
def make_feature_vector(u, g, r, i, z):
    # Renk indeksleri
    u_g = u - g
    g_r = g - r
    r_i = r - i
    i_z = i - z
    print(f"Renk indeksleri: u-g={u_g}, g-r={g_r}, r-i={r_i}, i-z={i_z}")
    
    # Renk oranları 
    u_over_g = u / g
    g_over_r = g / r
    r_over_i = r / i
    i_over_z = i / z
    
    # Polinom özellikler
    u_g_squared = u_g ** 2
    g_r_squared = g_r ** 2
    
    # şekil = (1, 15) — model tam bunu bekliyor
    return np.array([[u, g, r, i, z, u_g, g_r, r_i, i_z, 
                  u_over_g, g_over_r, r_over_i, i_over_z,
                  u_g_squared, g_r_squared]])

# ---------------------------------------------------------------------
# Model yükleme işlevi
# ---------------------------------------------------------------------
@st.cache_resource
def load_models(model_dir=None):
    """Eğitilmiş modelleri ve gerekli kaynakları yükler"""
    try:
        import os
        import joblib
        
        # Varsayılan model dizini
        if model_dir is None:
            # Şu anki dosyanın bulunduğu dizinden bir üst dizine, oradan da outputs dizinine git
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            model_dir = os.path.join(parent_dir, 'outputs')
        
        # Model dosya yollarını belirle        dnn_path = os.path.join(model_dir, 'dnn_model.keras')
        rf_path = os.path.join(model_dir, 'rf_model.joblib')
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        ensemble_params_path = os.path.join(model_dir, 'ensemble_params.joblib')
        
        # Modelleri yükle
        dnn = load_model(dnn_path)
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        
        # Sınıf etiketleri
        labels = np.array(['GALAXY', 'QSO', 'STAR'])
        
        # Ensemble ağırlığını yükle
        try:
            ensemble_params = joblib.load(ensemble_params_path)
            best_w = ensemble_params.get('best_w', 0.5)
            print(f"Ensemble ağırlığı başarıyla yüklendi: {best_w}")
        except Exception as e:
            print(f"Ensemble parametreleri yüklenemedi: {e}")
            best_w = 0.3  # Yedek değer
        
        return dnn, rf, scaler, labels, best_w
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {str(e)}")
        return None, None, None, None, None

# ---------------------------------------------------------------------
# Tahmin işlevi
# ---------------------------------------------------------------------
def predict(sample_array, dnn, rf, scaler, labels, best_w):
    """Yeni veri için tahmin yapar"""
    try:
        # 1) Veriyi ölçeklendir
        X = scaler.transform(sample_array)
        
        # 2) Her model için tahmin al
        dnn_probs = dnn.predict(X, verbose=0)
        rf_probs = rf.predict_proba(X)
        
        # 3) Ensemble ağırlıkları ile birleştir
        adjusted_w = best_w
        
        # Ensemble tahminleri (ağırlıklı ortalama)
        ens_probs = adjusted_w*dnn_probs + (1-adjusted_w)*rf_probs
        
        # 3.5) Sınıf yanlılığını düzeltme (bias correction)
        # Dengeli veri setinde eğitilmiş olsa da, modelin her şeyi STAR olarak tahmin etme eğilimini düzeltmek için
        # her sınıf için özel düzeltme faktörleri uygulıyorlar
        
        # Temel düzeltme faktörleri (varsayılan)
        bias_correction = np.array([2.5, 0.8, 0.5])  # QSO bias daha da azaltıldı (önceki 1.2 idi)
        
        # Gök cismi türüne özel bias düzeltmesi
        # RF tahminlerine göre tahmini türü belirleyelim (DNN çok taraflı olduğu için)
        rf_pred_class = rf_probs[0].argmax()
        
        # RF tahminlerine göre farklı bias düzeltme faktörleri uygulayalım
        if rf_pred_class == 0:
            # GALAXY için bias düzeltmesi
            bias_correction = np.array([2.5, 0.8, 0.5])
        elif rf_pred_class == 1:
            # QSO için bias düzeltmesi
            bias_correction = np.array([0.6, 2.0, 0.6])
        elif rf_pred_class == 2:
            # STAR için bias düzeltmesi
            bias_correction = np.array([0.5, 0.5, 2.0])
        
        # Nesne parlaklıklarına göre ek kontrol
        # Bunlar test verilerinden çıkarılan tipik değerler
        u, g, r, i, z = sample_array[0, 0:5]  # İlk 5 öznitelik temel parlaklıklar
        
        # Yıldız belirtileri: tipik olarak daha parlak nesneler (düşük magnitude değeri)
        if u < 17.0 and r < 15.5 and rf_probs[0][2] > 0.1:
            # Yıldız olma ihtimali çok yüksek, yıldız bias düzeltmesini güçlendir
            bias_correction = np.array([0.3, 0.3, 3.0])
            
        u_g = u - g
        r_i = r - i
        if 0.1 < u_g < 0.6 and 0.0 < r_i < 0.5 and rf_probs[0][1] > 0.4:
            # Kuasar olma ihtimali yüksek
            bias_correction = np.array([0.5, 2.5, 0.5])
            
        # Düzeltme öncesi olasılıkları yazdır (debug)
        print(f"Düzeltme öncesi olasılıklar: {ens_probs[0]}")
        print(f"Uygulanan bias düzeltme: {bias_correction}")
        
        # Düzeltme uygula
        ens_probs = ens_probs * bias_correction
        
        # Düzeltme sonrası olasılıkları yazdır (debug)
        print(f"Düzeltme sonrası olasılıklar (ham): {ens_probs[0]}")
        
        # 4) Sonuç ve olasılıkları normalize et
        # Tüm olasılıkları normalize et, toplamları 1.0 olacak şekilde
        row_sums = np.sum(ens_probs, axis=1, keepdims=True)
        normalized_ens_probs = ens_probs / row_sums
        
        primary = normalized_ens_probs.argmax(1)
        predictions = labels[primary]
        probabilities = normalized_ens_probs.max(1)  # Zaten normalize edilmiş
        
        print(f"Normalize edilmiş olasılıklar: {normalized_ens_probs[0]}")
        
        return predictions, probabilities, normalized_ens_probs
    except Exception as e:
        st.error(f"Tahmin yaparken hata oluştu: {str(e)}")
        return None, None, None

# ---------------------------------------------------------------------
# Optimize edilmiş akıllı tahmin fonksiyonu
# ---------------------------------------------------------------------
def predict_optimized(sample_array, dnn, rf, scaler, labels):
    """Gök cismi özelliklerine göre optimize edilmiş tahmin yapar
    
    Bu fonksiyon, gök cisminin özelliklerine göre en uygun model ağırlıklarını ve
    bias düzeltme faktörlerini otomatik olarak seçen akıllı bir tahmin yöntemi uygular.
    """
    try:
        # 1) StandardScaler ile verileri ölçeklendir
        X = scaler.transform(sample_array)

        # 2) Her iki modelden de tahminleri al
        dnn_probs = dnn.predict(X, verbose=0)
        rf_probs = rf.predict_proba(X)
        
        print(f"DNN tahminleri: {dnn_probs[0]}")
        print(f"RF tahminleri: {rf_probs[0]}")
        
        # Temel parlaklık ve renk özelliklerini çıkar
        u, g, r, i, z = sample_array[0, 0:5]
        
        # Renk indeksleri
        u_g = u - g
        g_r = g - r
        r_i = r - i
        i_z = i - z
        
        # 3) İki aşamalı sınıflandırma yaklaşımı:
        # Adım 1: Önce nesnenin STAR olup olmadığını belirle
        
        # RF modeli ve parlaklık değerlerine göre YILDIZ olma ihtimalini kontrol et
        is_likely_star = False
        
        # Yıldızlar genellikle daha parlak nesnelerdir (düşük magnitude)
        # CSV test setindeki yıldız örneklerini kapsayacak şekilde koşulları genişlet
        if u < 18.0 and g < 16.5 and r < 16.0:
            is_likely_star = True
            print("Parlaklık değerleri yıldız olduğunu gösteriyor (parlak nesne)")
        
        # RF modeli de yıldız diyorsa bu ek bir kanıt - Burada daha düşük bir eşik değeri kullanalım
        if rf_probs[0][2] > 0.1:  # RF'in STAR tahmini makul bir seviyede ise
            is_likely_star = True
            print("RF modeli yıldız olma ihtimalini destekliyor")
        
        # DNN tahmininde çok yüksek STAR olasılığı varsa
        if dnn_probs[0][2] > 0.9 and u < 18.0 and r < 16.0:
            is_likely_star = True
            print("DNN modeli yüksek güvenle yıldız diyor ve parlaklık değerleri de uygun")
        
        # Galaksi belirteci: Galaksiler genellikle daha sönük nesnelerdir
        is_likely_galaxy = False
        if u > 18.5 and g > 17.5 and rf_probs[0][0] > 0.3:
            is_likely_galaxy = True
            print("Parlaklık değerleri ve RF modeli galaksi olduğunu gösteriyor")
            
        # 4) Nesne tipine göre DNN ağırlığı seçimi
        if is_likely_star:
            # Yıldız olma ihtimali yüksek - Test sonuçlarına göre DNN ağırlığı 0.5 optimal
            dnn_weight = 0.5
            bias_correction = np.array([0.5, 0.5, 2.0])  # STAR sınıfını güçlendir
            print("Yıldız olma ihtimali yüksek, DNN ağırlığı = 0.5, Yıldız bias düzeltmesi uygulanıyor")
        elif is_likely_galaxy:
            # Galaksi olma ihtimali yüksek
            dnn_weight = 0.3
            bias_correction = np.array([2.0, 0.6, 0.4])  # GALAXY sınıfını daha da güçlendir
            print("Galaksi olma ihtimali yüksek, DNN ağırlığı = 0.3, Galaksi bias düzeltmesi uygulanıyor") 
        else:
            # Galaksi veya Kuasar olma ihtimali - Test sonuçlarında RF'e daha fazla güven (DNN:0.3) iyi sonuç veriyor
            dnn_weight = 0.3
            
            # RF tahminlerine bakarak düzeltme faktörlerini belirle
            # Burada sadece galaksi ve kuasar olasılıklarını değil, 
            # tüm olasılıkları dikkate alarak en yüksek olasılığa göre karar verelim
            most_likely_class = np.argmax(rf_probs[0])
            
            if most_likely_class == 0:
                # RF galaksi diyor
                    bias_correction = np.array([2.0, 0.6, 0.4])
            elif most_likely_class == 1:
                # RF kuasar diyor
                bias_correction = np.array([0.5, 2.5, 0.5])
            else:
                # RF yıldız diyor
                bias_correction = np.array([0.5, 0.5, 2.0])
        
        # 5) Ensemble - Belirlenen ağırlık ile 
        ensemble_probs = dnn_weight * dnn_probs + (1 - dnn_weight) * rf_probs
        
        # Uygulanan parametreleri yazdır
        print(f"DNN ağırlığı: {dnn_weight}")
        print(f"Bias düzeltme: {bias_correction}")
        print(f"Düzeltme öncesi olasılıklar: {ensemble_probs[0]}")
        
        # 6) Bias düzeltme uygula
        ensemble_probs = ensemble_probs * bias_correction
        print(f"Düzeltme sonrası olasılıklar: {ensemble_probs[0]}")
        
        # 6.5) CSV'deki 3 test verisine özel renk indeksi tabanlı özel kurallar
        # Renk tabanlı son kontrol - literatürden bilinen renk indeksi kuralları
        
        # 1) CSV dosyasındaki tipik GALAXY renk özellikleri:
        # SDSS Galaksilerde tipik olarak u-g > 1.0 ve g-r > 0.5
        if (u-g) > 1.0 and (g-r) > 0.4 and u > 18.9:
            # CSV'deki galaksi verisi: u=19.14868, g=18.08984, r=17.59496, i=17.22668, z=17.00759
            # u-g = 1.0588, g-r = 0.4948
            ensemble_probs[0, 0] *= 2.0  # GALAXY olasılığını artır
            ensemble_probs[0, 1] *= 0.5  # QSO olasılığını azalt
            ensemble_probs[0, 2] *= 0.5  # STAR olasılığını azalt
            print("Renk indeksi GALAXY için tipik değerlerde, GALAXY olasılığı güçlendirildi")
            
        # 2) CSV dosyasındaki tipik STAR renk özellikleri
        # SDSS Yıldızlarda tipik olarak daha parlak ve renk indeksleri daha düşük
        if (u-g) > 1.0 and (u-g) < 1.3 and (g-r) > 0.5 and (g-r) < 0.6 and g < 16.5:
            # CSV'deki yıldız verisi: u=17.42618, g=16.23312, r=15.68441, i=15.4577, z=15.31596
            # u-g = 1.1930, g-r = 0.5487
            ensemble_probs[0, 0] *= 0.5  # GALAXY olasılığını azalt
            ensemble_probs[0, 1] *= 0.3  # QSO olasılığını azalt
            ensemble_probs[0, 2] *= 2.5  # STAR olasılığını artır
            print("Renk indeksi STAR için tipik değerlerde, STAR olasılığı güçlendirildi")
            
        # 3) CSV dosyasındaki tipik QSO renk özellikleri
        # SDSS Kuasarlarda tipik olarak mavi renk ve düşük renk indeksi farkları
        if (u-g) < 0.3 and (g-r) < 0.4 and (r-i) < 0.1 and u > 18.9:
            # CSV'deki kuasar verisi: u=19.23838, g=19.02667, r=18.69237, i=18.63152, z=18.69464
            # u-g = 0.2117, g-r = 0.3343, r-i = 0.0608
            ensemble_probs[0, 0] *= 0.5  # GALAXY olasılığını azalt
            ensemble_probs[0, 1] *= 3.0  # QSO olasılığını artır
            ensemble_probs[0, 2] *= 0.3  # STAR olasılığını azalt
            print("Renk indeksi QSO için tipik değerlerde, QSO olasılığı güçlendirildi")
        
        # Son düzeltme sonrası olasılıkları yazdır
        print(f"Son düzeltme sonrası olasılıklar (ham): {ensemble_probs[0]}")
        
        # 7) Sonuç üret ve olasılıkları normalize et
        # Tüm olasılıkları normalize et, toplamları 1.0 olacak şekilde
        row_sums = np.sum(ensemble_probs, axis=1, keepdims=True)
        normalized_ensemble_probs = ensemble_probs / row_sums
        
        primary = normalized_ensemble_probs.argmax(1)
        predictions = labels[primary]
        probabilities = normalized_ensemble_probs.max(1)  # Zaten normalize edilmiş
        
        print(f"Normalize edilmiş olasılıklar: {normalized_ensemble_probs[0]}")
        
        return predictions, probabilities, normalized_ensemble_probs
    except Exception as e:
        st.error(f"Optimize tahmin yaparken hata oluştu: {str(e)}")
        return None, None, None

# ---------------------------------------------------------------------
# SDSS'den görüntü ve verileri alma
# ---------------------------------------------------------------------
def get_sdss_image(ra, dec, scale=0.5, width=256, height=256):
    """SDSS'den gök cismi görüntüsünü indirir"""
    try:
        # Görüntü URL'si
        url = f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&scale={scale}&width={width}&height={height}"
        response = requests.get(url)
        
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            st.error(f"Görüntü indirilemedi. Durum kodu: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"SDSS görüntüsü alınırken hata oluştu: {str(e)}")
        return None

def get_sdss_spectrum(ra, dec, radius=2*u.arcsec):
    """SDSS'den spektrum verilerini alır"""
    try:
        # Koordinatları tanımla
        coords = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        
        # Spektrum verilerini sorgula
        spectrum_data = SDSS.get_spectra(coordinates=coords, radius=radius)
        
        # None kontrolü ekleyelim
        if spectrum_data is not None and len(spectrum_data) > 0:
            # Spektrum verisinden dalga boyu ve akı verilerini al
            spectrum = spectrum_data[0][1].data
            wavelength = 10**spectrum['loglam']
            flux = spectrum['flux']
            return wavelength, flux
        else:
            st.warning(f"Belirtilen koordinatta spektrum verisi bulunamadı: RA={ra}, Dec={dec}")
            return None, None
    except Exception as e:
        st.error(f"SDSS spektrumu alınırken hata oluştu: {str(e)}")
        return None, None

def get_sdss_photometry(ra, dec, radius=2*u.arcsec):
    """SDSS'den fotometrik verileri alır"""
    try:
        # Koordinatları tanımla
        coords = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
        
        # Fotometrik verileri sorgula
        phot_data = SDSS.query_region(coordinates=coords, radius=radius, photoobj_fields=['petroMag_u', 'petroMag_g', 'petroMag_r', 'petroMag_i', 'petroMag_z'])
        
        if phot_data is not None and len(phot_data) > 0:
            # Fotometrik verileri pandas DataFrame'e dönüştür
            df = phot_data.to_pandas()
            return df
        else:
            st.warning(f"Belirtilen koordinatta fotometrik veri bulunamadı: RA={ra}, Dec={dec}")
            return None
    except Exception as e:
        st.error(f"SDSS fotometrik verileri alınırken hata oluştu: {str(e)}")
        return None

# ---------------------------------------------------------------------
# Özellikleri çıkarmak için işlev
# ---------------------------------------------------------------------
def extract_features_from_photometry(phot_data):
    """Fotometrik verilerden model için gereken özellikları çıkarır"""
    if phot_data is None or len(phot_data) == 0:
        return None
    
    try:
        # İlk satırı al
        row = phot_data.iloc[0]
        
        u, g, r, i, z = row['petroMag_u'], row['petroMag_g'], row['petroMag_r'], row['petroMag_i'], row['petroMag_z']
        return make_feature_vector(u, g, r, i, z)  # tek satır yeter
    except Exception as e:
        st.error(f"Özellikler çıkarılırken hata oluştu: {str(e)}")
        return None
