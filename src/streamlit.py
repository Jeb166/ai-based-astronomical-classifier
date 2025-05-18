#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st

# UI başlığı ve açıklaması - En başta olmalı
st.set_page_config(
    page_title="Astronomik Sınıflandırıcı",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
from prediction import get_sdss_object_by_objid

# Add styling after page config
page_bg_img = '''
<style>
body {
    background-image: url("https://www.example.com/background.jpg");
    background-size: cover;
    background-attachment: fixed;
    color: #ffffff;
}
.sidebar .sidebar-content {
    background: rgba(0, 0, 0, 0.7);
    color: #ffffff;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

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

def pick_band(row, band):
    """
    SDSS Row / Series içinden istenen bandın magnitüdünü bulur.
    Tercih sırası:
      u, psfMag_u, modelMag_u, cModelMag_u, fiberMag_u, petroMag_u (+büyük harf)
    """
    prefixes = ["", "psfMag_", "modelMag_", "cModelMag_", "fiberMag_", "petroMag_"]
    cand_cols = [p + band for p in prefixes] + [band.upper()]
    for c in cand_cols:
        if c in row and pd.notna(row[c]):
            return float(row[c])
    return None



# UI başlığı ve açıklaması
st.title("🌌 AI Tabanlı Astronomik Gök Cismi Sınıflandırıcı")
st.markdown("""
<div style="background-color:rgba(0, 0, 0, 0.7); padding: 10px; border-radius: 5px;">
    <p style="font-size: 18px; color: #ffffff;">
        Bu uygulama, Random Forest algoritması kullanarak astronomik gök cisimlerini sınıflandırır. 
        SDSS verilerini kullanarak galaksi, kuasar ve yıldız tespiti yapabilirsiniz.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Ana UI yapısı
# ---------------------------------------------------------------------
# Yan panel (sidebar) oluşturma
st.sidebar.header("Gök Cismi Araştırma")
st.sidebar.markdown("SDSS veri tabanını kullanarak gök cismi sınıflandırması yapın.")

# Giriş metodu seçimi
input_method = st.sidebar.radio(
    "Giriş metodu seçin:",
    ["Gökyüzü Haritası ile Arama", "Manuel Filtreleme Değerleri", "CSV Dosyası Yükleme"]
)

debug_mode = st.sidebar.checkbox("Debug modu")

def debug(*args, **kwargs):
    """Debug mod açıksa ekrana basar."""
    if debug_mode:
        st.write(*args, **kwargs)


# Modeli yükle
with st.spinner("Random Forest modeli yükleniyor..."):
    rf, scaler, labels = load_models()

if rf is not None and scaler is not None:
    st.sidebar.success("Random Forest modeli başarıyla yüklendi! 🚀")
      # ---------------------------------------------------------
    # Koordinat ile arama
    # ---------------------------------------------------------
    if input_method == "Gökyüzü Haritası ile Arama":
        
        st.subheader("Gökyüzü Haritası ile Gök Cismi Ara")        
        st.markdown("##### SDSS DR18 Navigasyon Aracı")
        st.markdown("SDSS'in Aladin Sky Atlas kullanan navigasyon aracıyla daha detaylı inceleme yapabilirsiniz.")          
        sdss_iframe = """
        <iframe id="naviframe" scrolling="yes" allow="clipboard-write" 
                style="width: 100%; height: 750px; overflow: visible; border:1px solid #ccc; border-radius:5px; background-color: #fff;" 
                src="https://skyserver.sdss.org/navigate/?ra=180&dec=0&scale=0.3&dr=18&opt=&embedded=true"></iframe>
        """
        components.html(sdss_iframe, height=800)

        # ObjID ve Koordinat ile arama için sekmeler oluştur
        tab1, tab2 = st.tabs(["ObjID ile Ara", "Koordinat ile Ara"])

        with tab1:
            st.markdown("### ObjID ile Gök Cismi Ara")
            objid_input = st.text_input("SDSS Nesne ID (objID)", placeholder="Örnek: 1237668296598749280")

            if st.button("ObjID ile Ara ve Sınıflandır", key="objid_search"):
                if not objid_input.strip():
                    st.warning("Lütfen bir ObjID girin.")
                    st.stop()

                with st.spinner("SDSS’ten veri çekiliyor..."):
                    try:
                        row = None
                        for dr in (18, 17, 16):
                            st.write(f"ObjID sorgulanıyor, DR={dr}")
                            try:
                                # get_sdss_object_by_objid içinde kullanılan istek URL'sini de görmek için debug
                                st.write("Sorgu yapılacak URL: <fonksiyondan_alınan_url>")  # Örnek debug
                                data = get_sdss_object_by_objid(objid_input.strip(), dr=dr)
                                st.write(f"DR={dr} için fonksiyon döndürdüğü veri: {data}")
                                if data is not None:
                                    row = data
                                    st.write(f"Veri bulundu: {row}")
                                    break
                                else:
                                    st.warning(f"Bu DR={dr} için kayıt bulunamadı. ObjID, DR={dr} kapsamı dışında olabilir.")
                            except Exception as e:
                                st.error(f"DR{dr} sorgusu hata verdi: {e}")
                    except Exception as e:
                        st.error(f"SDSS servisine bağlanılamadı: {e}")
                        st.stop()

                if row is None:
                    st.error("Bu ObjID için hiçbir DR sürümünde kayıt bulunamadı. ObjID’nin geçerli olduğundan emin olun veya farklı bir örnek deneyin.")
                    st.stop()

                # 2-c) Feature-vector oluştur
                u_  = pick_band(row, "u")
                g_  = pick_band(row, "g")
                r_  = pick_band(row, "r")
                i_  = pick_band(row, "i")
                z_  = pick_band(row, "z")

                missing = [b for b,v in zip("ugriz", (u_,g_,r_,i_,z_)) if v is None]
                if missing:
                    st.error(f"Bu nesnede fotometrik veri eksik ({', '.join(missing)} bandı yok). "
                            "Yarıçapı büyütüp farklı bir nesne deneyin.")
                    st.stop()

                fv = make_feature_vector(
                        u_, g_, r_, i_, z_,
                        plate   = row.get("plate",   0),
                        mjd     = row.get("mjd",     0),
                        fiberid = row.get("fiberid", 0),
                        redshift= row.get("redshift",0)
                )


                # Ölçekle + tahmin et
                X_scaled, _ = preprocess_data(fv, scaler)
                pred_class, confidence, class_probs = predict(X_scaled, rf, scaler, labels)

                # === Sonuçları göster ===
                st.success(f"Tahmin: **{pred_class}**  —  Güven: **{confidence:.3f}**")
                st.markdown(get_object_info_text(pred_class, confidence))

                col1, col2 = st.columns(2)
                col1.pyplot(plot_predictions(pred_class, class_probs))
                col2.pyplot(display_confidence_gauge(confidence))

                # İsteğe bağlı: gökyüzü görüntüsü
                img = get_sdss_image(row["ra"], row["dec"], scale=0.3)
                if img:
                    st.image(img, caption="SDSS kesiti")
                else:
                    st.warning("Lütfen bir ObjID girin.")

        with tab2:
            st.markdown("### Koordinat ile Gök Cismi Ara")
            col1, col2 = st.columns(2)
            with col1:
                ra = st.number_input("Sağ Açıklık (RA)", min_value=0.0, max_value=360.0, value=180.0, format="%.6f")
            with col2:
                dec = st.number_input("Dik Açıklık (Dec)", min_value=-90.0, max_value=90.0, value=0.0, format="%.6f")

            search_radius = st.slider("Arama Yarıçapı (derece)", 0.001, 0.05, 0.01, step=0.001, format="%.3f")

            if st.button("Koordinatlar ile Ara ve Sınıflandır", key="coord_search"):
                with st.spinner("SDSS’ten veri çekiliyor…"):
                    # 2-a) Koordinata en yakın nesneyi çek
                    radius_arcsec = search_radius * 3600        # derece → ″
                    row = get_sdss_object_by_coords(ra, dec,
                                                    radius_arcsec=radius_arcsec)
                    if row is None:
                        st.error("SkyServer yanıt vermedi veya bu koordinatta ugriz verisi yok. "
                                "• İnternet/VPN engelini kontrol edin\n"
                                "• Yarıçapı büyütüp yeniden deneyin")
                        st.stop()

                if row is None:
                    st.error("Bu koordinatlarda nesne bulunamadı; yarıçapı büyütmeyi deneyin.")
                    st.stop()

                # 2-b) Astropy Row → pandas.Series
                import pandas as pd

                if debug_mode:                # Sidebar’daki "Debug modu" kutusu açıkken
                    st.write("Gelen satır:", row)
                    if isinstance(row, pd.Series):
                        st.write("Sütun adları:", row.index.tolist())
                    else:   # astropy Row
                        st.write("Sütun adları:", row.colnames)

                 # 2-c) Feature-vector oluştur
                u_  = pick_band(row, "u")
                g_  = pick_band(row, "g")
                r_  = pick_band(row, "r")
                i_  = pick_band(row, "i")
                z_  = pick_band(row, "z")

                missing = [b for b,v in zip("ugriz", (u_,g_,r_,i_,z_)) if v is None]
                if missing:
                    st.error(f"Fotometrik veri eksik: {', '.join(missing)} band(lar)ı yok.\n"
                            "• Aynı koordinatta farklı objeler olabilir.\n"
                            "• Yarıçapı büyütüp başka bir nesne seçin **ya da** u/g/r/i/z "
                            "alanı içeren PhotoObj kaydını SQL ile çağırın.")
                    st.stop()

                fv = make_feature_vector(
                        u_, g_, r_, i_, z_,
                        plate   = row.get("plate",   0),
                        mjd     = row.get("mjd",     0),
                        fiberid = row.get("fiberid", 0),
                        redshift= row.get("redshift",0)
                )


                # 2-d) Ölçekle + tahmin et
                X_scaled, _ = preprocess_data(fv, scaler)
                pred_class, confidence, class_probs = predict(X_scaled, rf, scaler, labels)

                # 2-e) Sonuçları göster
                st.success(f"Tahmin: **{pred_class}**  —  Güven: **{confidence:.3f}**")
                st.markdown(get_object_info_text(pred_class, confidence))

                col1, col2 = st.columns(2)
                col1.pyplot(plot_predictions(pred_class, class_probs))
                col2.pyplot(display_confidence_gauge(confidence))

                # 2-f) SDSS görüntüsü
                img = get_sdss_image(row["ra"], row["dec"], scale=0.3)
                if img:
                    st.image(img, caption="SDSS kesiti")

                
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
                st.error(f"CSV dosyası işlenirken hata oluştu: {str(e)}")
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
