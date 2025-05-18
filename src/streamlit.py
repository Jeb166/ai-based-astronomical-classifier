#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import base64
from PIL import Image
from io import BytesIO
import streamlit.components.v1 as components
import urllib.parse as ul
import requests
import os
import time

# Model işlevlerini ve tahmin işlevlerini içe aktar
from prediction import (
    load_models, predict, get_sdss_image, get_sdss_object_by_coords,
    make_feature_vector, plot_predictions, display_confidence_gauge, 
    get_object_info_text, get_spectra_link
)

# ---------------------------------------------------------------------
# Veri ön işleme fonksiyonu (test_rf.py'den adapte edildi)
# ---------------------------------------------------------------------
def preprocess_data(df, scaler, debug=False):
    """CSV verilerini RF modeli için ön işler ve scaler ile uyumlu hale getirir"""
    try:
        if debug:
            st.write(f"Ön işleme öncesi veri boyutu: {df.shape}")
        
        # Eğer verinin kopyasını oluşturmamışsak, oluştur
        df = df.copy()
        
        # Koordinat ve ID sütunlarını kaldır
        cols_to_drop = []
        for col in ['objid', 'specobjid', 'run', 'rerun', 'camcol', 'field', 'ra', 'dec']:
            if col in df.columns:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            df = df.drop(cols_to_drop, axis=1)
            if debug:
                st.write(f"Kaldırılan sütunlar: {cols_to_drop}")
        
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
            if debug:
                st.warning(f"Sayısal olmayan sütunlar kaldırılıyor: {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)
        
        # NaN ve sonsuz değerleri kontrol et
        nan_count = df.isna().sum().sum()
        inf_count = ((df == np.inf) | (df == -np.inf)).sum().sum()
        if nan_count > 0 or inf_count > 0:
            if debug:
                st.warning(f"Eksik değerler bulundu: {nan_count} NaN, {inf_count} sonsuz değer")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(df.median(), inplace=True)
        
        # Scaler'ın özellik adlarını al
        if hasattr(scaler, 'feature_names_in_'):
            scaler_columns = set(scaler.feature_names_in_)
            
            # Özellik sütunlarını scaler'da olan sütunlarla eşleştir
            feature_columns = set(df.columns)
            
            # Eksik sütunları kontrol et
            missing_columns = scaler_columns - feature_columns
            if missing_columns:
                if debug:
                    st.warning(f"Modelin beklediği bazı sütunlar eksik: {missing_columns}")
                # Eksik sütunlar için 0 ile doldur
                for col in missing_columns:
                    df[col] = 0
            
            # Fazla sütunları kontrol et
            extra_columns = feature_columns - scaler_columns
            if extra_columns:
                if debug:
                    st.warning(f"Modelde olmayan fazla sütunlar kaldırılıyor: {extra_columns}")
                df = df.drop(columns=extra_columns)
            
            # Sütun sıralamasını scaler ile uyumlu hale getir
            df = df[scaler.feature_names_in_]
        else:
            if debug:
                st.warning("Scaler'da feature_names_in_ özelliği bulunamadı. Sütun uyumluluğu kontrol edilemiyor.")
        
        # Veriyi ölçeklendir
        X = scaler.transform(df)
        
        if debug:
            st.write(f"Ön işleme sonrası özellik vektörü boyutu: {X.shape}")
        
        return X, y
        
    except Exception as e:
        st.error(f"Veri ön işleme sırasında hata: {str(e)}")
        return None, None

# En yakın SDSS objesini bulmak için yardımcı fonksiyon
def query_nearest_obj(ra, dec, radius=0.01):
    """
    Verilen koordinatlara yakın gök cisimlerini araştırır.
    
    Parameters:
        ra (float): Sağ açıklık (derece)
        dec (float): Dik açıklık (derece)
        radius (float): Arama yarıçapı (derece); 0.01° ≈ 36 açı saniyesi
        
    Returns:
        pandas.DataFrame: Bulunan gök cisimlerinin verileri
    """    # Farklı SDSS DR API URL'lerini deneyelim - HTTP 500 hatası durumunda alternatif API'ler kullanılacak
    urls = [
        # DR18 en son sürüm (navigasyon sayfasından)
        f"https://skyserver.sdss.org/dr18/SkyServer/SearchTools/RadialSearch?ra={ra}&dec={dec}&radius={radius}&format=json",
        
        # DR17 ve DR16 yedek sürümler
        f"https://skyserver.sdss.org/dr17/SkyServer/SearchTools/RadialSearch?ra={ra}&dec={dec}&radius={radius}&format=json",
        f"https://skyserver.sdss.org/dr16/SkyServer/SearchTools/RadialSearch?ra={ra}&dec={dec}&radius={radius}&format=json",
        
        # Web servisi API'leri
        f"http://skyserver.sdss.org/dr16/SkyServerWS/SearchTools/RadialSearch?ra={ra}&dec={dec}&radius={radius}&format=json",
        f"http://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/RadialSearch?ra={ra}&dec={dec}&radius={radius}&format=json",
        f"http://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/RadialSearch?ra={ra}&dec={dec}&radius={radius}&format=json",
        
        # Alternatif API uçnoktaları
        f"https://skyserver.sdss.org/dr16/SkyServer/Search/RadialSearch?format=json&ra={ra}&dec={dec}&radius={radius}",
        f"https://skyserver.sdss.org/dr17/SkyServer/Search/RadialSearch?format=json&ra={ra}&dec={dec}&radius={radius}",
        f"https://skyserver.sdss.org/dr18/SkyServer/Search/RadialSearch?format=json&ra={ra}&dec={dec}&radius={radius}",
        
        # CasJobs SQL sorguları
        f"http://skyserver.sdss.org/CasJobs/RestApi/contexts/default/query?query=SELECT+TOP+10+*+FROM+PhotoObj+WHERE+CONTAINS(POINT('J2000',ra,dec),CIRCLE('J2000',{ra},{dec},{radius}))+ORDER+BY+distance&format=json"
    ]    # User-Agent ekleyerek istek başlıklarını hazırla
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/javascript, */*',
        'Origin': 'https://skyserver.sdss.org',
        'Referer': 'https://skyserver.sdss.org/navigate/'
    }
    
    last_error = None
    
    # Tüm API URL'lerini sırayla deneyelim
    for url in urls:
        # Debug için URL'yi yazdır
        print(f"SDSS API URL deneniyor: {url}")
        
        try:
            # Daha uzun bir timeout süresi ile isteği gönder
            response = requests.get(url, headers=headers, timeout=60)
            
            if response.status_code == 200:
                # Yanıtı göster ve debug
                content_type = response.headers.get('content-type', '')
                print(f"API yanıt content-type: {content_type}")
                
                try:
                    js = response.json()
                    
                    # Yanıt yapısını debug için yazdır
                    print(f"API yanıt yapısı: {type(js)}")
                    if isinstance(js, dict):
                        print(f"API yanıt anahtarları: {js.keys()}")
                    
                    # API yanıtı başarılı ancak veri boş olabilir
                    if js is None or (isinstance(js, list) and len(js) == 0) or (isinstance(js, dict) and len(js) == 0):
                        print(f"Bu URL için veri bulunamadı: {url}")
                        continue
                    
                    # API yanıt formatını kontrol et
                    if isinstance(js, dict) and "error" in js:
                        print(f"SDSS API hatası: {js['error']}")
                        continue
                    
                    # Yanıtta 'Exception' anahtarı olması durumu
                    if isinstance(js, dict) and "Exception" in js:
                        print(f"SDSS API Exception: {js['Exception']}")
                        continue
                        
                    # Farklı yanıt formatları için kontrol
                    if isinstance(js, dict):
                        # Data anahtarına sahip yanıt formatı
                        if "Rows" in js:
                            objects = js["Rows"]
                            if len(objects) > 0:
                                print(f"API başarılı sonuç döndü: {len(objects)} nesne bulundu")
                                return pd.DataFrame(objects)
                            else:
                                print("API yanıtında Rows anahtarı var ancak içi boş.")
                                continue
                        elif "data" in js:
                            objects = js["data"]
                            if len(objects) > 0:
                                print(f"API başarılı sonuç döndü: {len(objects)} nesne bulundu")
                                return pd.DataFrame(objects)
                            else:
                                print("API yanıtında data anahtarı var ancak içi boş.")
                                continue
                        elif "Column1" in js:
                            # Bazı eski SDSS API versiyonları Column1, Column2... şeklinde döner
                            print("API Column1 formatında yanıt döndü")
                            return pd.DataFrame([js])
                        else:
                            # Anahtarları doğrudan sütun olarak kullan
                            print("API farklı bir formatta yanıt döndü, doğrudan anahtarlar kullanılıyor")
                            return pd.DataFrame([js])
                    elif isinstance(js, list):
                        # Doğrudan liste formatı
                        if len(js) > 0:
                            print(f"API başarılı liste yanıtı döndü: {len(js)} nesne")
                            return pd.DataFrame(js)
                        else:
                            print("API yanıtı boş liste içeriyor.")
                            continue
                    else:
                        print(f"API'dan beklenmeyen veri formatı alındı: {type(js)}")
                        continue
                except ValueError as json_err:
                    # JSON parse hatası durumunda ham yanıtı incele ve devam et
                    resp_text = response.text[:1000]  # İlk 1000 karakter
                    print(f"API yanıtı JSON olarak ayrıştırılamadı: {json_err}. Ham yanıt: {resp_text}...")
                    last_error = json_err
                    continue
            else:
                # HTTP hata kodunda bir sonraki API URL'yi dene
                print(f"API HTTP hata kodu: {response.status_code}")
                try:
                    error_content = response.text[:500]  # İlk 500 karakter
                    print(f"Hata yanıtı: {error_content}")
                except:
                    print("Hata yanıtı okunamadı")
                
                last_error = f"HTTP {response.status_code}"
                continue
                
        except Exception as e:            # İstek hatası, bir sonraki URL'yi dene
            print(f"API istek hatası: {str(e)}")
            last_error = e
            continue
    
    # Tüm URL'ler denendikten sonra hala başarısızsa, hata döndür
    error_msg = f"Tüm SDSS API URL'leri başarısız oldu. Son hata: {str(last_error) if last_error else 'Bilinmeyen hata'}"
    st.error(error_msg)
    print(error_msg)
    
    # Kullanıcıya arama yarıçapını artırmasını öner
    if radius < 0.05:
        st.info(f"İpucu: Arama yarıçapını artırmayı deneyin (şu anki yarıçap: {radius} derece). Daha büyük bir arama alanında daha fazla gök cismi bulunabilir.")
    
    return None

# UI başlığı ve açıklaması
st.set_page_config(
    page_title="Astronomik Sınıflandırıcı",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Random Forest Tabanlı Astronomik Gök Cismi Sınıflandırıcı")
st.markdown("""
Bu uygulama, Random Forest algoritması kullanarak astronomik gök cisimlerini sınıflandırır. 
SDSS verilerini kullanarak galaksi, kuasar ve yıldız tespiti yapabilirsiniz.
""")

# ---------------------------------------------------------------------
# Ana UI yapısı
# ---------------------------------------------------------------------
# Yan panel (sidebar) oluşturma
st.sidebar.header("Gök Cismi Araştırma")
st.sidebar.markdown("SDSS veri tabanını kullanarak gök cismi sınıflandırması yapın.")

# Giriş metodu seçimi
input_method = st.sidebar.radio(
    "Giriş metodu seçin:",
    ["Koordinat ile Arama", "Manuel Filtreleme Değerleri", "CSV Dosyası Yükleme", "Örnek Veriler"]
)

# Modeli yükle
with st.spinner("Random Forest modeli yükleniyor..."):
    rf, scaler, labels = load_models()

if rf is not None and scaler is not None:
    st.sidebar.success("Random Forest modeli başarıyla yüklendi! 🚀")
      # ---------------------------------------------------------
    # Koordinat ile arama
    # ---------------------------------------------------------
    if input_method == "Koordinat ile Arama":
        st.subheader("Koordinat ile Gök Cismi Ara")        
        st.markdown("### SDSS DR18 Navigasyon Aracı")
        st.markdown("SDSS'in orijinal navigasyon aracıyla daha detaylı inceleme yapabilirsiniz.")
        
        sdss_iframe = """
        <iframe id="naviframe" scrolling="no" allow="clipboard-write" 
                style="width: 100%; overflow: hidden; border:1px solid #ccc; border-radius:5px; background-color: #fff;" 
                height="550" 
                src="https://skyserver.sdss.org/navigate/?ra=180&dec=0&scale=0.3&dr=18&opt=&embedded=true"></iframe>
        <script>
        // Koordinatları ana sayfaya göndermek için mesaj dinleyicisi ekleyelim
        window.addEventListener('message', function(event) {
            if (event.data && event.data.type === 'coordinates') {
                // Streamlit'e özel mesaj gönderme mekanizması
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: event.data.coordinates
                }, '*');
            }
        });
        </script>
        """
        components.html(sdss_iframe, height=570)
        st.info("İpucu: Haritada RA ve Dec koordinatlarını öğrenmek için mouse ile bir noktaya tıkladıktan sonra sağ üstte görünen koordinat bilgilerini kullanabilirsiniz. Koordinat sistemini sol üstte **ICRSd** olarak ayarlayın.")

        col1, col2 = st.columns(2)
        with col1:
            ra = st.number_input("Sağ Açıklık (RA)", min_value=0.0, max_value=360.0, value=180.0, format="%.6f")
        with col2:
            dec = st.number_input("Dik Açıklık (Dec)", min_value=-90.0, max_value=90.0, value=0.0, format="%.6f")

        search_radius = st.slider("Arama Yarıçapı (derece)", 0.001, 0.05, 0.01, step=0.001, format="%.3f")        
        if st.button("Ara ve Sınıflandır", key="search_coords_fixed"):
            with st.spinner("SDSS'ten en yakın gök cismi aranıyor..."):
                results_df = query_nearest_obj(ra, dec, search_radius)

            if results_df is not None and not results_df.empty:
                st.success(f"{len(results_df)} gök cismi bulundu.")
                
                # Veri önizlemesi - ilk 10 satır
                with st.expander("Bulunan gök cisimlerinin verileri", expanded=True):
                    st.dataframe(results_df.head(10))

                # En yakın cismi seç
                closest_obj = results_df.iloc[0]
                
                # objID veya objid (büyük/küçük harf farklılıkları olabilir)
                objid = closest_obj.get('objID') or closest_obj.get('objid')
                
                if objid:
                    st.info(f"En yakın gök cismi: ID = {objid}")
                      # SDSS nesnesinden detaylı bilgileri al
                    with st.spinner("Detaylı SDSS verisi alınıyor..."):
                        detailed = get_sdss_object_by_coords(closest_obj['ra'], closest_obj['dec'])
                    
                    # Görüntü alma
                    st.markdown("### SDSS Görüntüsü")
                    with st.spinner("Görüntü alınıyor..."):
                        image = get_sdss_image(closest_obj['ra'], closest_obj['dec'])
                        if image:
                            st.image(image, caption=f"RA: {closest_obj['ra']}, Dec: {closest_obj['dec']}", width=400)
                        else:
                            st.warning("SDSS görüntüsü alınamadı")

                    # Detaylı bilgi varsa sınıflandır
                    if detailed is not None:
                        try:
                            st.markdown("### Fotometrik Veriler")
                            col1, col2, col3, col4, col5 = st.columns(5)
                            
                            # u, g, r, i, z değerleri
                            u = float(detailed["u"]) if "u" in detailed.colnames else float(closest_obj.get('u', 0))
                            g = float(detailed["g"]) if "g" in detailed.colnames else float(closest_obj.get('g', 0))
                            r = float(detailed["r"]) if "r" in detailed.colnames else float(closest_obj.get('r', 0))
                            i_ = float(detailed["i"]) if "i" in detailed.colnames else float(closest_obj.get('i', 0))
                            z = float(detailed["z"]) if "z" in detailed.colnames else float(closest_obj.get('z', 0))
                            
                            # Değerleri göster
                            with col1:
                                st.metric(label="u filtresi", value=f"{u:.3f}")
                            with col2:
                                st.metric(label="g filtresi", value=f"{g:.3f}")
                            with col3:
                                st.metric(label="r filtresi", value=f"{r:.3f}")
                            with col4:
                                st.metric(label="i filtresi", value=f"{i_:.3f}")
                            with col5:
                                st.metric(label="z filtresi", value=f"{z:.3f}")
                            
                            # Spektral veriler
                            plate = detailed["plate"] if "plate" in detailed.colnames else closest_obj.get('plate', 0)
                            mjd = detailed["mjd"] if "mjd" in detailed.colnames else closest_obj.get('mjd', 0)
                            fiberid = detailed["fiberid"] if "fiberid" in detailed.colnames else closest_obj.get('fiberid', 0)
                            redshift = detailed["z"] if "z" in detailed.colnames else closest_obj.get('redshift', 0)
                            
                            if plate and mjd and fiberid:
                                st.markdown("### Spektral Veriler")
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(label="Plate ID", value=plate)
                                with col2:
                                    st.metric(label="MJD", value=mjd)
                                with col3:
                                    st.metric(label="Fiber ID", value=fiberid)
                                with col4:
                                    st.metric(label="Redshift", value=f"{redshift:.6f}" if redshift else "N/A")
                                
                                # Spektrum bağlantısı
                                spec_obj = {'plate': plate, 'mjd': mjd, 'fiberid': fiberid}
                                spec_link = get_spectra_link(spec_obj)
                                if spec_link:
                                    st.markdown(f"[SDSS Spektrum Görüntüleyicide Aç]({spec_link})")
                            
                            # Özellik vektörü oluştur
                            sample = make_feature_vector(u, g, r, i_, z, 
                                                        plate=plate, 
                                                        mjd=mjd, 
                                                        fiberid=fiberid, 
                                                        redshift=redshift)
                            
                            # Sınıflandırma yap
                            st.markdown("### Sınıflandırma Sonucu")
                            with st.spinner("Sınıflandırma yapılıyor..."):
                                # Özellik vektörünü hazırla (scaler öncesi)
                                X, _ = preprocess_data(sample, scaler, debug=False)
                                if X is not None:
                                    # Tahmin yap
                                    pred_class, confidence, class_probs = predict(X, rf, scaler, labels)
                                    
                                    # Sonuçları göster
                                    st.subheader(f"Tahmin Edilen Sınıf: {pred_class}")
                                    st.markdown(f"**Güven Değeri:** {confidence:.4f}")
                                    
                                    # Görselleştirme
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.pyplot(plot_predictions(pred_class, class_probs))
                                    with col2:
                                        st.pyplot(display_confidence_gauge(confidence))
                                    
                                    # Açıklayıcı bilgi
                                    st.info(get_object_info_text(pred_class, confidence))
                                else:
                                    st.error("Veri ön işleme başarısız oldu.")
                                
                        except Exception as e:
                            st.error(f"Sınıflandırma sırasında hata oluştu: {str(e)}")
                    else:
                        st.warning("SDSS objesi bulundu ama detaylı veriler alınamadı. Spektral verileri olmayan bir obje olabilir.")
                else:
                    st.error("Bulunan gök cisminde objID bilgisi bulunmuyor.")
            else:                st.warning("Bu koordinatlarda SDSS verisi bulunamadı. Yarıçapı artırmayı veya başka bir bölgeyi deneyin.")
        else:
            st.warning("Bu koordinatlarda SDSS verisi bulunamadı. Yarıçapı artırmayı veya başka bir bölgeyi deneyin.")

        
    # ---------------------------------------------------------
    # Manuel Filtreleme Değerleri
    # ---------------------------------------------------------
    elif input_method == "Manuel Filtreleme Değerleri":
        st.subheader("Fotometrik Değerlerle Manuel Sınıflandırma")        
        st.markdown("""
        SDSS'in beş temel fotometrik filtreleme değerlerini (u, g, r, i, z) girerek sınıflandırma yapabilirsiniz.
        Değerleri kadir (magnitude) cinsinden giriniz.
        """)        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            u_mag = st.number_input("u filtresi (kadir)", min_value=10.0, max_value=30.0, value=19.0, format="%.5f")
        with col2:
            g_mag = st.number_input("g filtresi (kadir)", min_value=10.0, max_value=30.0, value=17.5, format="%.5f")
        with col3:
            r_mag = st.number_input("r filtresi (kadir)", min_value=10.0, max_value=30.0, value=16.8, format="%.5f")
        with col4:
            i_mag = st.number_input("i filtresi (kadir)", min_value=10.0, max_value=30.0, value=16.5, format="%.5f")
        with col5:
            z_mag = st.number_input("z filtresi (kadir)", min_value=10.0, max_value=30.0, value=16.2, format="%.5f")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            plate = st.number_input("Plate ID", min_value=0, max_value=20000, value=1000)
        with col2:
            mjd = st.number_input("MJD (Modified Julian Date)", min_value=50000, max_value=60000, value=55000)
        with col3:
            fiberid = st.number_input("Fiber ID", min_value=0, max_value=1000, value=500)
        with col4:
            redshift = st.number_input("Redshift (z)", min_value=-1.0, max_value=10.0, value=0.1, format="%.10f")
        
        if st.button("Sınıflandır", key="manual_classify"):
            try:
                # Özellik vektörü oluştur
                sample_df = make_feature_vector(
                    u_mag, g_mag, r_mag, i_mag, z_mag,
                    plate=plate, mjd=mjd, fiberid=fiberid, redshift=redshift
                )

                # Debug bilgisi
                st.write(f"Debug - Oluşturulan DataFrame: şekil={sample_df.shape}, tip={type(sample_df)}")

                # Tahmini yap
                with st.spinner("Sınıflandırma yapılıyor..."):
                    # preprocess_data ile CSV bölümünde kullanılan aynı yöntemi kullan
                    X_scaled, _ = preprocess_data(sample_df, scaler, debug=True)
                    
                    if X_scaled is None:
                        st.error("Veri ön işleme başarısız oldu.")
                    else:
                        # Model ile tahmin
                        rf_probs = rf.predict_proba(X_scaled)
                        pred_classes_idx = rf_probs.argmax(1)
                        pred_class = labels[pred_classes_idx[0]]
                        confidence = rf_probs[0, pred_classes_idx[0]]
                        
                        # Tüm sınıf olasılıklarını hazırla
                        class_probs = {label: float(rf_probs[0, i]) for i, label in enumerate(labels)}
                      # Sonuçları göster
                    st.subheader(f"Sınıflandırma Sonucu: {pred_class}")
                    st.markdown(f"**Güven Değeri:** {confidence:.4f}")
                    
                    # Tahmin olasılıklarını göster
                    st.write("Tahmin olasılıkları:")
                    for cls, prob in class_probs.items():
                        st.write(f"{cls}: {prob:.4f}")
                    
                    # Açıklama ekle
                    st.markdown(get_object_info_text(pred_class, confidence))
                    
                    # Renk indekslerini göster
                    st.subheader("Renk İndeksleri")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("u - g", f"{(u_mag - g_mag):.2f}")
                    col2.metric("g - r", f"{(g_mag - r_mag):.2f}")
                    col3.metric("r - i", f"{(r_mag - i_mag):.2f}")
                    col4.metric("i - z", f"{(i_mag - z_mag):.2f}")
                    
                    # Grafik göster
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(plot_predictions(pred_class, class_probs))
                    with col2:
                        st.pyplot(display_confidence_gauge(confidence))
                    
            except Exception as e:
                st.error(f"Sınıflandırma hatası: {str(e)}")
    # ---------------------------------------------------------
    # CSV Dosyası Yükleme
    # ---------------------------------------------------------
    elif input_method == "CSV Dosyası Yükleme":
        st.subheader("CSV Dosyası ile Toplu Sınıflandırma")
        st.markdown("""
        CSV dosyası yükleyerek birden fazla gök cismini toplu olarak sınıflandırabilirsiniz.
        
        CSV dosyanızda en azından `u`, `g`, `r`, `i`, `z` sütunları bulunmalıdır. Opsiyonel olarak `class` sütunu 
        eklerseniz, tahminlerin doğruluğunu değerlendirebilirsiniz.
        """)
        
        uploaded_file = st.file_uploader("CSV dosyası yükleyin", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)                
                st.write("CSV dosyası yüklendi! İlk birkaç satır:")
                st.dataframe(df.head())
                
                # Gerekli sütunların olup olmadığını kontrol et
                required_cols = ['u', 'g', 'r', 'i', 'z']
                if all(col in df.columns for col in required_cols):
                    show_debug = st.checkbox("Hata ayıklama bilgilerini göster", value=False)
                    if st.button("Toplu Sınıflandır", key="batch_classify"):                        
                        with st.spinner("Sınıflandırma yapılıyor... Bu biraz zaman alabilir."):
                            try:
                                # test_rf.py'de kullanılan preprocess_data fonksiyonunu kullan
                                X_scaled, true_classes = preprocess_data(df, scaler, debug=show_debug)
                                
                                if X_scaled is None:
                                    st.error("Veri ön işleme başarısız oldu.")
                                else:
                                    # Tahmin yap
                                    start_time = time.time()
                                    rf_probs = rf.predict_proba(X_scaled)
                                    
                                    # Sonuçları çıkar
                                    pred_classes_idx = rf_probs.argmax(1)
                                    pred_classes = [labels[idx] for idx in pred_classes_idx]
                                    confidences = [rf_probs[i, idx] for i, idx in enumerate(pred_classes_idx)]
                                    
                                    # Sonuçları DataFrame'e ekle
                                    results_df = df.copy()
                                    results_df['predicted_class'] = pred_classes
                                    results_df['confidence'] = confidences
                                    
                                    # Sonuçları göster
                                    st.success(f"Sınıflandırma tamamlandı! ({time.time() - start_time:.2f} saniye)")
                                    st.dataframe(results_df)
                                    
                                    # İstatistikler
                                    st.subheader("Sınıflandırma İstatistikleri")
                                    
                                    # Sınıf dağılımı
                                    class_dist = pd.Series(pred_classes).value_counts()
                                    st.bar_chart(class_dist)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.metric("Ortalama Güven", f"{np.mean(confidences):.4f}")
                                    with col2:
                                        st.metric("Medyan Güven", f"{np.median(confidences):.4f}")
                                    
                                    # Gerçek değerler ile karşılaştırma
                                    has_class = 'class' in df.columns and true_classes is not None
                                    if has_class:
                                        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
                                        
                                        accuracy = accuracy_score(true_classes, pred_classes)
                                        st.metric("Doğruluk (Accuracy)", f"{accuracy:.4f}")
                                        
                                        st.subheader("Sınıflandırma Raporu")
                                        report = classification_report(true_classes, pred_classes, output_dict=True)
                                        report_df = pd.DataFrame(report).transpose()
                                        st.dataframe(report_df)
                                        
                                        st.subheader("Karmaşıklık Matrisi")
                                        cm = confusion_matrix(true_classes, pred_classes)
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                                    xticklabels=labels,
                                                    yticklabels=labels)
                                        plt.title('Karmaşıklık Matrisi')
                                        plt.xlabel('Tahmin Edilen Sınıf')
                                        plt.ylabel('Gerçek Sınıf')
                                        st.pyplot(fig)
                                    
                                    # Sonuçları CSV olarak indirme
                                    csv = results_df.to_csv(index=False)
                                    b64 = base64.b64encode(csv.encode()).decode()
                                    href = f'<a href="data:file/csv;base64,{b64}" download="siniflandirma_sonuclari.csv">Sonuçları CSV Olarak İndir</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Sınıflandırma sırasında hata oluştu: {str(e)}")
                else:
                    missing = [col for col in required_cols if col not in df.columns]
                    st.error(f"CSV dosyasında gerekli sütunlar eksik: {', '.join(missing)}")
            except Exception as e:
                st.error(f"CSV dosyası işlenirken hata oluştu: {str(e)}")# ---------------------------------------------------------
    # Örnek Veriler
    # ---------------------------------------------------------
    elif input_method == "Örnek Veriler":
        st.subheader("Örnek Gök Cisimleri ile Test Et")
        st.markdown("""
        Test etmek için aşağıdaki örnek gök cisimlerinden birini seçin.
        Bu örnekler, modelleri test etmek için kullanılan SDSS veri setinden alınmıştır.
        """)
        
        # Örnek objeler
        examples = {
            "Galaksi Örneği": {
                "u": 19.149, "g": 18.090, "r": 17.595, "i": 17.272, "z": 17.146,
                "class": "GALAXY", "ra": 344.544, "dec": -0.245
            },
            "Kuasar (QSO) Örneği": {
                "u": 19.238, "g": 19.027, "r": 18.692, "i": 18.584, "z": 18.269,
                "class": "QSO", "ra": 333.384, "dec": 2.390
            },
            "Yıldız Örneği": {
                "u": 17.426, "g": 16.233, "r": 15.684, "i": 15.444, "z": 15.332,
                "class": "STAR", "ra": 249.170, "dec": 22.276
            },
            "Belirsiz Örnek (Zorlayıcı)": {
                "u": 20.547, "g": 19.801, "r": 19.519, "i": 19.064, "z": 18.969,
                "class": "QSO", "ra": 321.650, "dec": 10.121
            }
        }
        
        selected_example = st.selectbox("Örnek seç", list(examples.keys()))
        
        if st.button("Seçilen Örneği Sınıflandır", key="classify_example"):
            example = examples[selected_example]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Gök Cismi Bilgileri")
                st.write(f"**Sağ Açıklık (RA):** {example['ra']}")
                st.write(f"**Dik açıklık (Dec):** {example['dec']}")
                st.write(f"**Gerçek Sınıf:** {example['class']}")
                
                # Filtreleme değerlerini göster
                st.markdown("### Fotometrik Değerler (Magnitude)")
                col_a, col_b, col_c, col_d, col_e = st.columns(5)
                col_a.metric("u", f"{example['u']:.3f}")
                col_b.metric("g", f"{example['g']:.3f}")
                col_c.metric("r", f"{example['r']:.3f}")
                col_d.metric("i", f"{example['i']:.3f}")
                col_e.metric("z", f"{example['z']:.3f}")
                
                # Renk indekslerini göster
                st.markdown("### Renk İndeksleri")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("u - g", f"{(example['u'] - example['g']):.2f}")
                col_b.metric("g - r", f"{(example['g'] - example['r']):.2f}")
                col_c.metric("r - i", f"{(example['r'] - example['i']):.2f}")
                col_d.metric("i - z", f"{(example['i'] - example['z']):.2f}")
            
            with col2:
                # SDSS görüntüsünü göster
                st.markdown("### SDSS Görüntüsü")
                with st.spinner("Görüntü alınıyor..."):
                    image = get_sdss_image(example['ra'], example['dec'])
                    if image:
                        st.image(image, caption=f"RA: {example['ra']}, Dec: {example['dec']}", width=400)                    
                    else:
                        st.warning("Görüntü alınamadı")
                        
            # Sınıflandırma yap
            # Opsiyonel parametreleri örnek veri içinde varsa al, yoksa varsayılan değerleri kullan
            plate_val = example.get('plate', 0)
            mjd_val = example.get('mjd', 0)
            fiberid_val = example.get('fiberid', 0)
            redshift_val = example.get('redshift', 0)
            
            sample_df = make_feature_vector(
                example['u'], 
                example['g'], 
                example['r'], 
                example['i'], 
                example['z'],
                plate=plate_val,
                mjd=mjd_val,
                fiberid=fiberid_val,
                redshift=redshift_val
            )
            
            with st.spinner("Sınıflandırma yapılıyor..."):
                # preprocess_data ile CSV bölümünde kullanılan aynı yöntemi kullan
                X_scaled, _ = preprocess_data(sample_df, scaler, debug=False)
                
                if X_scaled is None:
                    st.error("Veri ön işleme başarısız oldu.")
                else:
                    # Model ile tahmin
                    rf_probs = rf.predict_proba(X_scaled)
                    pred_classes_idx = rf_probs.argmax(1)
                    pred_class = labels[pred_classes_idx[0]]
                    confidence = rf_probs[0, pred_classes_idx[0]]
                    
                    # Tüm sınıf olasılıklarını hazırla
                    class_probs = {label: float(rf_probs[0, i]) for i, label in enumerate(labels)}
                
                # Sonuçları göster
                st.markdown("---")
                st.subheader("Sınıflandırma Sonucu")
                
                # Sonuç ile gerçek değeri karşılaştır
                accuracy = "✓ Doğru" if pred_class == example['class'] else "✗ Yanlış"
                accuracy_color = "green" if pred_class == example['class'] else "red"
                
                st.markdown(f"**Tahmin:** {pred_class} &nbsp; **Gerçek:** {example['class']} &nbsp; **Sonuç:** <span style='color:{accuracy_color}'>{accuracy}</span>", unsafe_allow_html=True)
                st.markdown(f"**Güven Değeri:** {confidence:.4f}")
                
                # Açıklama ekle
                st.markdown(get_object_info_text(pred_class, confidence))
                
                # Grafik göster
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_predictions(pred_class, class_probs))
                with col2:
                    st.pyplot(display_confidence_gauge(confidence))
else:
    st.error("Random Forest modeli yüklenemedi. Lütfen model dosyalarını kontrol edin.")

# ---------------------------------------------------------------------
# Hakkında bölümü
# ---------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Hakkında")
st.sidebar.markdown("""
Bu uygulama, bir Random Forest modeli kullanarak SDSS fotometrik verilerinden
göksel cisimleri (Galaksi, QSO/Kuasar, Yıldız) sınıflandırır.

Model, SDSS DR18 veri setindeki 100.000+ örnek ile eğitilmiştir.
""")
