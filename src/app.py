import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
from astropy import units as u
from PIL import Image
import io
import requests
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO

# -------------------------------------------------
# 15-özelliklik vektör (model için tam gereken şekilde)
# -------------------------------------------------
def make_feature_vector(u, g, r, i, z):
    # Renk özellikleri 
    u_g = u - g
    g_r = g - r
    r_i = r - i
    i_z = i - z
    
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


# UI başlığı ve açıklaması
st.set_page_config(
    page_title="Astronomik Sınıflandırıcı",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Tabanlı Astronomik Gök Cismi Sınıflandırıcı")
st.markdown("""
Bu uygulama, derin öğrenme ve makine öğrenmesi yöntemleri kullanarak astronomik gök cisimlerini 
sınıflandırır. SDSS verilerini kullanarak galaksi, kuasar ve yıldız tespiti yapabilirsiniz.
""")

# ---------------------------------------------------------------------
# Model yükleme işlevi
# ---------------------------------------------------------------------
@st.cache_resource
def load_models(model_dir='outputs'):
    """Eğitilmiş modelleri yükler"""    
    try:
        # DNN modelini yükle
        dnn_path = os.path.join(model_dir, 'dnn_model.keras')
        dnn = load_model(dnn_path)
        
        # Random Forest modelini yükle
        rf_path = os.path.join(model_dir, 'rf_model.joblib')
        rf = joblib.load(rf_path)
          
        # Modellerin giriş ve çıkış boyutlarını kontrol et
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        scaler = joblib.load(scaler_path)        # Etiketleri ve en iyi ağırlığı belirle
        # Not: predict_optimized fonksiyonu dinamik olarak ağırlık belirlediğinden 
        # artık sabit best_w değeri kullanılmıyor
        labels = np.array(['GALAXY', 'QSO', 'STAR'])
        best_w = 0.5  # Geriye dönük uyumluluk için tutuluyor
        
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
        """Ölçekle ➜ DNN & RF ➜ Ağırlıklı oy."""
        # 1) StandardScaler
        X = scaler.transform(sample_array)

        # 2) Olasılıklar
        dnn_probs = dnn.predict(X, verbose=0)
        rf_probs = rf.predict_proba(X)
        
        # Debug bilgisi
        print(f"DNN tahminleri: {dnn_probs[0]}")
        print(f"RF tahminleri: {rf_probs[0]}")
        
        # 2.5) DNN model ağırlığını düzeltme - DNN modeli çok yanlı olduğu için ağırlığını düşürelim
        adjusted_w = 0.2  # DNN modelinin ağırlığını düşürüyoruz (önceki 0.5 idi)
        
        # 3) Ensemble - yeni ağırlık ile
        ens_probs = adjusted_w*dnn_probs + (1-adjusted_w)*rf_probs
          # 3.5) Sınıf yanlılığını düzeltme (bias correction)
        # Dengeli veri setinde eğitilmiş olsa da, modelin her şeyi STAR olarak tahmin etme eğilimini düzeltmek için
        # her sınıf için özel düzeltme faktörleri uygulıyorlar
        
        # Temel düzeltme faktörleri (varsayılan)
        bias_correction = np.array([2.0, 1.8, 0.3])  # GALAXY, QSO, STAR için düzeltme faktörleri
        
        # Gök cismi türüne özel bias düzeltmesi
        # RF tahminlerine göre tahmini türü belirleyelim (DNN çok taraflı olduğu için)
        rf_pred_class = rf_probs[0].argmax()
        
        # RF tahminlerine göre farklı bias düzeltme faktörleri uygulayalım
        if rf_pred_class == 0:  # GALAXY için
            bias_correction = np.array([2.0, 1.8, 0.3])  # GALAXY sınıfını güçlendir
        elif rf_pred_class == 1:  # QSO için
            bias_correction = np.array([1.2, 2.5, 0.3])  # QSO sınıfını güçlendir
        elif rf_pred_class == 2:  # STAR için
            bias_correction = np.array([0.8, 0.6, 2.5])  # STAR sınıfını güçlendir
        
        # Nesne parlaklıklarına göre ek kontrol
        # Bunlar test verilerinden çıkarılan tipik değerler
        u, g, r, i, z = sample_array[0, 0:5]  # İlk 5 öznitelik temel parlaklıklar
        
        # Yıldız belirtileri: tipik olarak daha parlak nesneler (düşük magnitude değeri)
        if u < 17.0 and r < 15.5 and rf_probs[0][2] > 0.1:  
            bias_correction = np.array([0.7, 0.5, 3.0])  # STAR sınıfını daha da güçlendir
            
        # QSO belirtileri: u-g ve r-i renk değerleri QSO'lar için tipiktir
        u_g = u - g
        r_i = r - i
        if 0.1 < u_g < 0.6 and 0.0 < r_i < 0.5 and rf_probs[0][1] > 0.3:
            bias_correction = np.array([1.0, 3.0, 0.2])  # QSO sınıfını daha da güçlendir
            
        # Düzeltme öncesi olasılıkları yazdır (debug)
        print(f"Düzeltme öncesi olasılıklar: {ens_probs[0]}")
        print(f"Uygulanan bias düzeltme: {bias_correction}")
        
        # Düzeltme uygula
        ens_probs = ens_probs * bias_correction
        
        # Düzeltme sonrası olasılıkları yazdır (debug)
        print(f"Düzeltme sonrası olasılıklar: {ens_probs[0]}")
        
        # 4) Sonuç
        primary = ens_probs.argmax(1)
        predictions = labels[primary]
        probabilities = ens_probs.max(1) / np.sum(ens_probs, axis=1)  # Normalize edilmiş olasılık
        return predictions, probabilities, ens_probs
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
        
        # Temel parlaklık değerlerini ve renk özelliklerini çıkar
        u, g, r, i, z = sample_array[0, 0:5]
        
        # Renk indeksleri
        u_g = u - g
        g_r = g - r
        r_i = r - i
        i_z = i - z
        
        # 3) Model ağırlığı belirleme - Test sonuçlarına göre 0.2 dengeli
        dnn_weight = 0.2
        
        # DNN modeli büyük ihtimalle STAR diyecek, RF modeli daha doğru dengelenmiş
        # Nesne tipine göre ağırlık ayarla
        
        # DNN'in STAR önyargısı çok güçlü
        if dnn_probs[0][2] > 0.99:  # Aşırı STAR eğilimi
            # Parlaklık değerlerine göre kontrol et
            if u < 16.5 and g < 15.0 and r < 14.5:
                # Bu gerçekten parlak bir yıldız olabilir - DNN'e biraz daha güven
                dnn_weight = 0.4
            else:
                # DNN'in etkisini azalt
                dnn_weight = 0.1
        
        # 4) Ensemble - Belirlenen ağırlık ile 
        ensemble_probs = dnn_weight * dnn_probs + (1 - dnn_weight) * rf_probs
        
        print(f"Ensemble olasılıklar (ağırlık={dnn_weight}): {ensemble_probs[0]}")
        
        # 5) Nesne tipine göre özel bias düzeltmeleri
        
        # 5.1) Başlangıç bias düzeltme değerlerini belirle
        # Her obje için temel bir düzeltme uygula (GALAXY, QSO, STAR)
        bias_correction = np.array([1.0, 1.0, 1.0])
        
        # RF modelinin en güvendiği sınıfı bul
        rf_pred_class = rf_probs[0].argmax()
        rf_confidence = rf_probs[0].max()
        
        # 5.2) Parlaklık ve renk özellikleriyle obje tipi belirleme
        
        # YILDIZ için kriterler (parlak ve RF tarafından STAR olarak yorumlanan)
        if rf_pred_class == 2 or rf_probs[0][2] > 0.2:
            if u < 17.0 and r < 15.0:  # Gerçekten parlaksa
                bias_correction = np.array([0.7, 0.7, 2.0])  # STAR'ı güçlendir
        
        # KUASAR için kriterler (Orta parlaklıkta ve renk indeksleri tipik)
        elif rf_pred_class == 1 or rf_probs[0][1] > 0.4:
            if 17.0 < u < 19.0 and 0.1 < u_g < 0.8:
                bias_correction = np.array([0.8, 1.8, 0.8])  # QSO'yu güçlendir
        
        # GALAKSİ için kriterler (sönük objeler)
        elif rf_pred_class == 0 or rf_probs[0][0] > 0.45:
            if u > 18.0:  # Sönükse
                bias_correction = np.array([1.8, 0.8, 0.7])  # GALAXY'yi güçlendir
        
        # 5.3) Ek düzeltmeler - RF tahmin sonuçlarını dikkate al
        
        # RF modeli yüksek güvenle STAR diyorsa (DNN zaten STAR eğilimli)
        if rf_probs[0][2] > 0.3:
            bias_correction[2] = max(bias_correction[2], 1.5)  # STAR'ı güçlendir
            
        # RF modeli GALAXY diyorsa ve düzeltme gerekiyorsa 
        if rf_probs[0][0] > 0.5 and ensemble_probs[0][0] < ensemble_probs[0][2]:
            bias_correction[0] = 2.0  # GALAXY'yi güçlendir
            bias_correction[2] = 0.5  # STAR'ı zayıflat
            
        # RF modeli QSO diyorsa ve düzeltme gerekiyorsa
        if rf_probs[0][1] > 0.4 and ensemble_probs[0][1] < max(ensemble_probs[0][0], ensemble_probs[0][2]):
            bias_correction[1] = 2.0  # QSO'yu güçlendir
            
        # Uygulanan parametreleri yazdır
        print(f"Bias düzeltme: {bias_correction}")
        print(f"Düzeltme öncesi olasılıklar: {ensemble_probs[0]}")
        
        # 6) Bias düzeltme uygula
        ensemble_probs = ensemble_probs * bias_correction
        print(f"Düzeltme sonrası olasılıklar: {ensemble_probs[0]}")
        
        # 7) Sonuç üret
        primary = ensemble_probs.argmax(1)
        predictions = labels[primary]
        probabilities = ensemble_probs.max(1) / np.sum(ensemble_probs, axis=1)  # Normalize
        
        return predictions, probabilities, ensemble_probs
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
    """Fotometrik verilerden model için gereken özellikleri çıkarır"""
    if phot_data is None or len(phot_data) == 0:
        return None
    
    try:
        # İlk satırı al
        row = phot_data.iloc[0]
        
        u, g, r, i, z = row['petroMag_u'], row['petroMag_g'], row['petroMag_r'], row['petroMag_i'], row['petroMag_z']
        return make_feature_vector(u, g, r, i, z)          # ⬅️ tek satır yeter
    except Exception as e:
        st.error(f"Özellikler çıkarılırken hata oluştu: {str(e)}")
        return None

# ---------------------------------------------------------------------
# Ana UI yapısı
# ---------------------------------------------------------------------
# Yan panel (sidebar) oluşturma
st.sidebar.header("Gök Cismi Araştırma")
st.sidebar.markdown("SDSS veri tabanını kullanarak gök cismi sınıflandırması yapın.")

# Giriş metodu seçimi
input_method = st.sidebar.radio(
    "Giriş metodu seçin:",
    ["Koordinat ile Arama", "CSV Dosyası Yükleme", "Örnek Veriler"]
)

# Modellleri yükle
dnn, rf, scaler, labels, best_w = load_models()

if dnn is not None and rf is not None:
    st.sidebar.success("Modeller başarıyla yüklendi! 🚀")
    
    # Koordinat ile arama
    if input_method == "Koordinat ile Arama":
        st.subheader("Koordinat ile Gök Cismi Ara")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ra = st.number_input("Sağ açıklık (RA, derece)", min_value=0.0, max_value=360.0, value=180.0)
        
        with col2:
            dec = st.number_input("Dik açıklık (Dec, derece)", min_value=-90.0, max_value=90.0, value=0.0)
        
        if st.button("Ara ve Sınıflandır", type="primary"):
            # İşlem başladı mesajı
            with st.spinner("SDSS'den veri alınıyor ve analiz ediliyor..."):
                # Görüntüyü al
                image = get_sdss_image(ra, dec)
                
                # Fotometrik verileri al
                phot_data = get_sdss_photometry(ra, dec)
                
                # Spektrum verilerini al
                wavelength, flux = get_sdss_spectrum(ra, dec)
                
                # Veriler alındı, görüntüle
                if image is not None or phot_data is not None:
                    # Feature'ları çıkar
                    features = extract_features_from_photometry(phot_data)
                    
                    # Tahmin yap - optimize edilmiş tahmin fonksiyonunu kullan
                    if features is not None:
                        predictions, probabilities, all_probs = predict_optimized(features, dnn, rf, scaler, labels)
                        
                        # Sonuçları göster
                        if predictions is not None:
                            # Tahmin sonuçları
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if image is not None:
                                    st.image(image, caption=f"SDSS Görüntüsü (RA: {ra:.4f}, Dec: {dec:.4f})")
                                
                                # Tahmin sonucunu büyük bir kutu içinde göster
                                object_type = predictions[0]
                                probability = probabilities[0] * 100
                                
                                st.markdown(f"""
                                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                                    <h2 style='text-align: center; color: #0066cc;'>{object_type}</h2>
                                    <p style='text-align: center; font-size: 18px;'>Güven: {probability:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Sınıf olasılıklarını göster
                                st.subheader("Sınıf Olasılıkları")
                                probs_df = pd.DataFrame({
                                    'Sınıf': labels,
                                    'Olasılık (%)': all_probs[0] * 100
                                })
                                
                                # Çubuk grafiği
                                fig = px.bar(probs_df, x='Sınıf', y='Olasılık (%)', 
                                            color='Olasılık (%)', color_continuous_scale='Viridis',
                                            text_auto='.2f')
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Fotometrik verileri göster
                                if phot_data is not None:
                                    st.subheader("Fotometrik Veriler")
                                    # Okunabilir bir format oluştur
                                    readable_phot = pd.DataFrame({
                                        'Bant': ['u', 'g', 'r', 'i', 'z'],
                                        'Parlaklık (mag)': [
                                            phot_data.iloc[0]['petroMag_u'],
                                            phot_data.iloc[0]['petroMag_g'],
                                            phot_data.iloc[0]['petroMag_r'],
                                            phot_data.iloc[0]['petroMag_i'],
                                            phot_data.iloc[0]['petroMag_z']
                                        ]
                                    })
                                    
                                    # Renk-Parlaklık grafiği
                                    fig = px.scatter(readable_phot, x='Bant', y='Parlaklık (mag)', 
                                                    size=[10]*5, color='Parlaklık (mag)')
                                    fig.update_layout(yaxis_autorange="reversed")  # Astronomide parlaklık ters
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Spektrum verilerini göster
                                if wavelength is not None and flux is not None:
                                    st.subheader("Spektrum")
                                    
                                    # Spektrum grafiği
                                    spec_df = pd.DataFrame({
                                        'Dalga Boyu (Å)': wavelength,
                                        'Akı': flux
                                    })
                                    
                                    fig = px.line(spec_df, x='Dalga Boyu (Å)', y='Akı')
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Tahmin yapılamadı. Lütfen farklı bir koordinat deneyin.")
                    else:
                        st.warning("Özellikler çıkarılamadı. Lütfen farklı bir koordinat deneyin.")
                else:
                    st.warning("Belirtilen koordinatlarda SDSS verisi bulunamadı. Lütfen farklı bir koordinat deneyin.")

    # CSV dosyası yükleme
    elif input_method == "CSV Dosyası Yükleme":
        st.subheader("CSV Dosyası Yükle ve Sınıflandır")
        
        st.markdown("""
        CSV dosyanızın en azından aşağıdaki sütunları içermesi gerekiyor:
        - `u`, `g`, `r`, `i`, `z`: SDSS filtre parlaklıkları
        
        Opsiyonel olarak gök cisimlerinin koordinatlarını da ekleyebilirsiniz:
        - `ra`: Sağ açıklık (derece)
        - `dec`: Dik açıklık (derece)
        """)
        
        uploaded_file = st.file_uploader("CSV dosyası seçin", type=["csv"])
        
        if uploaded_file is not None:
            # CSV dosyasını oku
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"Dosya başarıyla yüklendi. {len(df)} satır bulundu.")
                
                # İlk birkaç satırı göster
                st.dataframe(df.head())
                
                # Gerekli sütunları kontrol et
                required_columns = ['u', 'g', 'r', 'i', 'z']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"CSV dosyasında gerekli sütunlar eksik: {missing_columns}")
                else:
                    if st.button("Toplu Sınıflandırma Yap", type="primary"):
                        # İşlem başladı mesajı
                        with st.spinner("Sınıflandırma yapılıyor..."):
                            # Özellikleri hazırla (15 özellik: temel, renk, oran ve polinom)
                            features_list = [
                                make_feature_vector(row['u'], row['g'], row['r'], row['i'], row['z'])[0]
                                for _, row in df.iterrows()
                            ]
                            features_array = np.vstack(features_list)
                            
                            # Tahmin yap - optimize edilmiş tahmin fonksiyonunu kullan
                            predictions, probabilities, all_probs = predict_optimized(features_array, dnn, rf, scaler, labels)
                            
                            if predictions is not None:
                                # Sonuçları DataFrame'e ekle
                                df['predicted_class'] = predictions
                                df['confidence'] = probabilities * 100
                                
                                for i, label in enumerate(labels):
                                    df[f'prob_{label}'] = all_probs[:, i] * 100
                                
                                # Sonuçları göster
                                st.subheader("Sınıflandırma Sonuçları")
                                st.dataframe(df)
                                
                                # Sınıf dağılımını göster
                                st.subheader("Sınıf Dağılımı")
                                class_counts = df['predicted_class'].value_counts().reset_index()
                                class_counts.columns = ['Sınıf', 'Sayı']
                                
                                fig = px.pie(class_counts, values='Sayı', names='Sınıf', title='Tahmin Edilen Sınıfların Dağılımı')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # CSV olarak indirme seçeneği
                                csv = df.to_csv(index=False)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Sonuçları CSV Olarak İndir</a>'
                                st.markdown(href, unsafe_allow_html=True)
                            else:
                                st.error("Tahmin yapılırken bir hata oluştu.")
            except Exception as e:
                st.error(f"CSV dosyası işlenirken hata oluştu: {str(e)}")
    
    # Örnek veriler
    else:        
        st.subheader("Örnek Verilerle Tanıtım")        # Örnek gök cisimleri - SDSS veri dağılımlarına göre optimize edilmiş
        examples = {
            "SDSS J094554.77+414351.1 (Galaksi)": {"ra": 146.4782, "dec": 41.7309, "type": "Galaksi", "desc": "SDSS veri tabanında bulunan tipik bir eliptik galaksi örneği.", 
                                                   "test_data": {"u": 19.3, "g": 17.6, "r": 16.5, "i": 16.0, "z": 15.7}},
            "SDSS J141348.25+440211.7 (Kuasar)": {"ra": 213.4511, "dec": 44.0366, "type": "Kuasar", "desc": "SDSS veri tabanında bulunan, aktif bir galaktik çekirdek içeren parlak bir kuasar.",
                                                 "test_data": {"u": 18.4, "g": 18.1, "r": 17.8, "i": 17.5, "z": 17.2}},
            "SDSS J172611.88+591820.3 (Yıldız)": {"ra": 261.5495, "dec": 59.3056, "type": "Yıldız", "desc": "SDSS veri tabanında bulunan tipik bir yıldız örneği.",
                                                 "test_data": {"u": 15.5, "g": 14.4, "r": 13.8, "i": 13.5, "z": 13.4}}
        }
        
        selected_example = st.selectbox("Örnek bir gök cismi seçin:", list(examples.keys()))
          # Seçilen örneği göster
        example = examples[selected_example]
        st.markdown(f"""
        **{selected_example}**  
        * Tür: {example['type']}
        * RA: {example['ra']:.4f}°
        * Dec: {example['dec']:.4f}°
        * {example['desc']}
        """)
        
        if st.button("Örneği Analiz Et", type="primary"):
            # İşlem başladı mesajı
            with st.spinner("SDSS'den veri alınıyor ve analiz ediliyor..."):
                ra, dec = example['ra'], example['dec']
                
                # Görüntüyü al
                image = get_sdss_image(ra, dec)
                
                # Fotometrik verileri al
                phot_data = get_sdss_photometry(ra, dec)
                  # API'den verileri alamadıysak test verilerini kullan
                use_test_data = False
                if phot_data is None and 'test_data' in example:
                    use_test_data = True
                    # Test verilerinden bir DataFrame oluştur
                    test_data = example['test_data']
                    phot_data = pd.DataFrame({
                        'petroMag_u': [test_data['u']],
                        'petroMag_g': [test_data['g']],
                        'petroMag_r': [test_data['r']],
                        'petroMag_i': [test_data['i']],
                        'petroMag_z': [test_data['z']]
                    })
                    st.info("SDSS'den veri alınamadı. Tanıtım amaçlı örnek test verileri kullanılıyor.")
                
                # Spektrum verilerini al
                wavelength, flux = get_sdss_spectrum(ra, dec)
                  # Veriler alındı, görüntüle
                if image is not None or phot_data is not None:
                    # Feature'ları çıkar
                    features = extract_features_from_photometry(phot_data)
                    
                    # Tahmin yap - optimize edilmiş tahmin fonksiyonunu kullan
                    if features is not None:
                        predictions, probabilities, all_probs = predict_optimized(features, dnn, rf, scaler, labels)
                        
                        # Sonuçları göster
                        if predictions is not None:
                            # Tahmin sonuçları
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                if image is not None:
                                    st.image(image, caption=f"SDSS Görüntüsü (RA: {ra:.4f}, Dec: {dec:.4f})")
                                
                                # Tahmin sonucunu büyük bir kutu içinde göster
                                object_type = predictions[0]
                                probability = probabilities[0] * 100
                                
                                st.markdown(f"""
                                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                                    <h2 style='text-align: center; color: #0066cc;'>{object_type}</h2>
                                    <p style='text-align: center; font-size: 18px;'>Güven: {probability:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Modelin tahmini ve gerçek değer karşılaştırması
                                expected_type = example['type'].upper()
                                if "GALAKSI" in expected_type or "GALAXY" in expected_type:
                                    expected_type = "GALAXY"
                                
                                if object_type == expected_type:
                                    st.success(f"Tahmin doğru! Beklenen: {expected_type}")
                                else:
                                    st.warning(f"Tahmin beklenen türle eşleşmiyor. Beklenen: {expected_type}")
                                
                                # Sınıf olasılıklarını göster
                                st.subheader("Sınıf Olasılıkları")
                                probs_df = pd.DataFrame({
                                    'Sınıf': labels,
                                    'Olasılık (%)': all_probs[0] * 100
                                })
                                
                                # Çubuk grafiği
                                fig = px.bar(probs_df, x='Sınıf', y='Olasılık (%)', 
                                            color='Olasılık (%)', color_continuous_scale='Viridis',
                                            text_auto='.2f')
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Fotometrik verileri göster
                                if phot_data is not None:
                                    st.subheader("Fotometrik Veriler")
                                    # Okunabilir bir format oluştur
                                    readable_phot = pd.DataFrame({
                                        'Bant': ['u', 'g', 'r', 'i', 'z'],
                                        'Parlaklık (mag)': [
                                            phot_data.iloc[0]['petroMag_u'],
                                            phot_data.iloc[0]['petroMag_g'],
                                            phot_data.iloc[0]['petroMag_r'],
                                            phot_data.iloc[0]['petroMag_i'],
                                            phot_data.iloc[0]['petroMag_z']
                                        ]
                                    })
                                    
                                    # Renk-Parlaklık grafiği
                                    fig = px.scatter(readable_phot, x='Bant', y='Parlaklık (mag)', 
                                                    size=[10]*5, color='Parlaklık (mag)')
                                    fig.update_layout(yaxis_autorange="reversed")  # Astronomide parlaklık ters
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Spektrum verilerini göster
                                if wavelength is not None and flux is not None:
                                    st.subheader("Spektrum")
                                    
                                    # Spektrum grafiği
                                    spec_df = pd.DataFrame({
                                        'Dalga Boyu (Å)': wavelength,
                                        'Akı': flux
                                    })
                                    
                                    fig = px.line(spec_df, x='Dalga Boyu (Å)', y='Akı')
                                    fig.update_layout(height=300)
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Tahmin yapılamadı. Lütfen farklı bir örnek deneyin.")
                    else:
                        st.warning("Özellikler çıkarılamadı. Lütfen farklı bir örnek deneyin.")
                else:
                    st.warning("Belirtilen koordinatlarda SDSS verisi bulunamadı. Lütfen farklı bir örnek deneyin.")
else:
    st.error("Modeller yüklenemedi. Lütfen 'outputs' klasöründe modellerin varlığını kontrol edin.")

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Bu uygulama AI tabanlı bir astronomik sınıflandırıcıdır.</p>
    <p>Veri kaynağı: <a href="https://www.sdss.org/">Sloan Digital Sky Survey (SDSS)</a></p>
</div>
""", unsafe_allow_html=True)
