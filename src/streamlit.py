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
    """
    url = (f"https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools"
           f"/RadialSearch?ra={ra}&dec={dec}&radius={radius}&format=json")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            js = response.json()
            return pd.DataFrame(js)  # objId, u,g,r,i,z vs. içeren DataFrame
        return None
    except Exception as e:
        st.error(f"SDSS radial arama hatası: {str(e)}")
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

        st.markdown("### Gökyüzü Haritası (Aladin Lite)")
        st.markdown("Haritada gezinin, ardından koordinatları manuel girin.")

        aladin_iframe = """
        <iframe src="https://aladin.u-strasbg.fr/AladinLite/?target=180+0&fov=0.2&survey=P/SDSS9/color"
                width="100%" height="500" style="border:1px solid #ccc; border-radius:5px;"></iframe>
        """
        components.html(aladin_iframe, height=520)

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
            st.dataframe(results_df)

            closest_obj = results_df.iloc[0]
            objid = closest_obj.get('objID') or closest_obj.get('objid')

            with st.spinner("Detaylı SDSS verisi alınıyor..."):
                detailed = get_sdss_object_by_coords(objid)

            if detailed:
                try:
                    u, g, r, i_, z = map(float, [detailed["u"], detailed["g"], detailed["r"], detailed["i"], detailed["z"]])
                    sample = make_feature_vector(u, g, r, i_, z)

                    with st.spinner("SDSS görüntüsü alınıyor..."):
                        image = get_sdss_image(closest_obj['ra'], closest_obj['dec'])
                        if image:
                            st.image(image, caption=f"RA: {closest_obj['ra']}, Dec: {closest_obj['dec']}", width=400)                    
                            with st.spinner("Sınıflandırma yapılıyor..."):
                                # CSV bölümünde kullanılan aynı sınıflandırma yöntemini kullan
                                X_scaled, _ = preprocess_data(sample, scaler, debug=False)

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
                            
                            st.subheader(f"Sınıflandırma Sonucu: {pred_class}")
                            st.markdown(f"**Güven Değeri:** {confidence:.4f}")
                            st.markdown(get_object_info_text(pred_class, confidence))

                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(plot_predictions(pred_class, class_probs))
                        with col2:
                            st.pyplot(display_confidence_gauge(confidence))

                except Exception as e:
                    st.error(f"Fotometrik veriler alınamadı veya dönüştürülemedi: {str(e)}")
            else:
                st.error("SDSS objesi bulundu ama detaylı veriler alınamadı. Belki taranmamış olabilir.")
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
