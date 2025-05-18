#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
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
        
        # Aladin Lite harita görüntüleyici
        st.markdown("### Etkileşimli Gökyüzü Haritası")
        st.markdown("Aşağıdaki haritada herhangi bir noktaya tıklayarak o koordinatları otomatik olarak seçebilirsiniz.")
        
        aladin_html = """
        <div style="text-align: center; width: 100%;">
            <div id="aladin-lite-div" style="height: 500px; width: 100%; border: 1px solid #ccc; border-radius: 5px; position: relative;"></div>
            <div id="status-message" style="margin-top: 10px; padding: 5px; color: #555; font-style: italic; background-color: #f8f9fa; border-radius: 3px;">Harita yükleniyor...</div>
        </div>
        <script type="text/javascript">
            // Aladin Lite haritayı yükle
            document.addEventListener("DOMContentLoaded", function() {
                if (typeof A === 'undefined') {
                    var statusElem = document.getElementById('status-message');
                    if (statusElem) statusElem.innerText = "Aladin Lite yüklenirken bekleyin...";
                    
                    var script = document.createElement('script');
                    script.src = 'https://aladin.u-strasbg.fr/AladinLite/api/v2/latest/aladin.min.js';
                    script.onload = initAladin;
                    document.head.appendChild(script);
                } else {
                    initAladin();
                }
            });

            function initAladin() {
                var statusElem = document.getElementById('status-message');
                if (statusElem) statusElem.innerText = "Gökyüzü haritası hazırlanıyor...";
                
                try {
                    // Varsayılan görünüm (SDSS görüntüleri için uygun bir alan)
                    var ra0 = 180.0;  // Sağ açıklık (RA)
                    var dec0 = 0.0;   // Dik açıklık (Dec)
                    
                    var aladin = A.aladin('#aladin-lite-div', {
                        survey: "P/SDSS9/color", 
                        fov: 0.2,
                        target: ra0 + " " + dec0,
                        showLayersControl: true,
                        showFullscreenControl: true,
                        showFrame: true,
                        showGotoControl: true
                    });
                    
                    // Katalog katmanı ekle (tıklanabilir objeler için)
                    var cat = A.catalog({name: 'Örnek Objeler'});
                    aladin.addCatalog(cat);
                    
                    // Haritada tıklama olayını yakala
                    aladin.on('objectClicked', function(object) {
                        var ra = object.ra;
                        var dec = object.dec;
                        
                        // Koordinatları güncellemek için Streamlit ile iletişim kur
                        window.parent.postMessage({
                            type: "streamlit:setComponentValue",
                            value: {ra: ra, dec: dec}
                        }, "*");
                        
                        if (statusElem) statusElem.innerText = "Seçilen koordinatlar: RA=" + ra.toFixed(6) + ", Dec=" + dec.toFixed(6);
                    });
                    
                    if (statusElem) statusElem.innerText = "Harita hazır! Bir noktaya tıklayarak koordinatları seçebilirsiniz.";
                    
                } catch (error) {
                    console.error("Aladin yüklenirken hata:", error);
                    if (statusElem) statusElem.innerText = "Harita yüklenemedi: " + error.message;
                }
            }
        </script>
        """
        
        components.html(aladin_html, height=600)
        
        col1, col2 = st.columns(2)
        with col1:
            ra = st.number_input("Sağ Açıklık (RA)", min_value=0.0, max_value=360.0, value=180.0, format="%.6f")
        with col2:
            dec = st.number_input("Dik Açıklık (Dec)", min_value=-90.0, max_value=90.0, value=0.0, format="%.6f")
        
        search_radius = st.slider("Arama Yarıçapı (derece)", min_value=0.001, max_value=0.05, value=0.01, step=0.001, format="%.3f")
        
        if st.button("Ara ve Sınıflandır", key="search_coords"):
            with st.spinner("SDSS'ten veriler alınıyor..."):
                try:
                    # En yakın objeleri bul
                    results_df = query_nearest_obj(ra, dec, search_radius)
                    
                    if results_df is not None and not results_df.empty:
                        st.success(f"{len(results_df)} adet gök cismi bulundu!")
                        
                        # Sonuçları göster
                        st.dataframe(results_df)
                        
                        # En yakın objeyi seç
                        closest_obj = results_df.iloc[0]
                        
                        # Görüntüyü al
                        with st.spinner("Gök cismi görüntüsü alınıyor..."):
                            image = get_sdss_image(closest_obj['ra'], closest_obj['dec'])
                            if image:
                                st.image(image, caption=f"RA: {closest_obj['ra']}, Dec: {closest_obj['dec']}", width=400)
                        
                        # Filtreleme verilerini hazırla
                        try:
                            sample = make_feature_vector(
                                float(closest_obj['u']), 
                                float(closest_obj['g']), 
                                float(closest_obj['r']), 
                                float(closest_obj['i']), 
                                float(closest_obj['z'])
                            )
                            
                            # Tahmini yap
                            with st.spinner("Sınıflandırma yapılıyor..."):
                                pred_class, confidence, class_probs = predict(sample, rf, scaler, labels)
                                
                                # Sonuçları göster
                                st.subheader(f"Sınıflandırma Sonucu: {pred_class}")
                                st.markdown(f"**Güven Değeri:** {confidence:.4f}")
                                
                                # Açıklama ekle
                                st.markdown(get_object_info_text(pred_class, confidence))
                                
                                # Grafik göster
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.pyplot(plot_predictions(pred_class, class_probs))
                                with col2:
                                    st.pyplot(display_confidence_gauge(confidence))
                                
                        except Exception as e:
                            st.error(f"Sınıflandırma hatası: {str(e)}")
                    else:
                        st.warning(f"Belirtilen koordinatlarda ({ra}, {dec}) gök cismi bulunamadı. Lütfen farklı koordinatlar deneyin veya arama yarıçapını arttırın.")
                except Exception as e:
                    st.error(f"Arama hatası: {str(e)}")
        
    # ---------------------------------------------------------
    # Manuel Filtreleme Değerleri
    # ---------------------------------------------------------
    elif input_method == "Manuel Filtreleme Değerleri":
        st.subheader("Fotometrik Değerlerle Manuel Sınıflandırma")
        st.markdown("""
        SDSS'in beş temel fotometrik filtreleme değerlerini (u, g, r, i, z) girerek sınıflandırma yapabilirsiniz.
        Değerleri kadir (magnitude) cinsinden giriniz.
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            u_mag = st.number_input("u filtresi (kadir)", min_value=10.0, max_value=30.0, value=19.0, format="%.4f")
            g_mag = st.number_input("g filtresi (kadir)", min_value=10.0, max_value=30.0, value=17.5, format="%.4f")
        with col2:
            r_mag = st.number_input("r filtresi (kadir)", min_value=10.0, max_value=30.0, value=16.8, format="%.4f")
            i_mag = st.number_input("i filtresi (kadir)", min_value=10.0, max_value=30.0, value=16.5, format="%.4f")
        with col3:
            z_mag = st.number_input("z filtresi (kadir)", min_value=10.0, max_value=30.0, value=16.2, format="%.4f")
        
        if st.button("Sınıflandır", key="manual_classify"):
            try:
                # Özellik vektörü oluştur
                sample = make_feature_vector(u_mag, g_mag, r_mag, i_mag, z_mag)
                
                # Tahmini yap
                with st.spinner("Sınıflandırma yapılıyor..."):
                    pred_class, confidence, class_probs = predict(sample, rf, scaler, labels)
                    
                    # Sonuçları göster
                    st.subheader(f"Sınıflandırma Sonucu: {pred_class}")
                    st.markdown(f"**Güven Değeri:** {confidence:.4f}")
                    
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
                    if st.button("Toplu Sınıflandır", key="batch_classify"):
                        with st.spinner("Sınıflandırma yapılıyor... Bu biraz zaman alabilir."):
                            # Sütunları çıkar
                            non_feature_cols = ['class', 'objid', 'specobjid', 'ra', 'dec', 'run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid', 'redshift']
                            feature_df = df.copy()
                            
                            # Eğer class sütunu varsa, değerlendirme için sakla
                            has_class = 'class' in feature_df.columns
                            true_classes = None
                            if has_class:
                                true_classes = feature_df['class'].copy()
                            
                            # Gerekli olmayan sütunları kaldır
                            for col in non_feature_cols:
                                if col in feature_df.columns:
                                    feature_df = feature_df.drop(col, axis=1)
                            
                            # Veri hazırlama (renk indeksleri ekleme)
                            if 'u_g' not in feature_df.columns:
                                feature_df['u_g'] = feature_df['u'] - feature_df['g']
                            if 'g_r' not in feature_df.columns:
                                feature_df['g_r'] = feature_df['g'] - feature_df['r']
                            if 'r_i' not in feature_df.columns:
                                feature_df['r_i'] = feature_df['r'] - feature_df['i']
                            if 'i_z' not in feature_df.columns:
                                feature_df['i_z'] = feature_df['i'] - feature_df['z']
                            
                            # Ölçeklendirme için diğer özellikler
                            if 'u_over_g' not in feature_df.columns:
                                feature_df['u_over_g'] = feature_df['u'] / feature_df['g']
                            if 'g_over_r' not in feature_df.columns:
                                feature_df['g_over_r'] = feature_df['g'] / feature_df['r']
                            if 'r_over_i' not in feature_df.columns:
                                feature_df['r_over_i'] = feature_df['r'] / feature_df['i']
                            if 'i_over_z' not in feature_df.columns:
                                feature_df['i_over_z'] = feature_df['i'] / feature_df['z']
                            
                            # Polinom özellikler
                            if 'u_g_squared' not in feature_df.columns:
                                feature_df['u_g_squared'] = feature_df['u_g'] ** 2
                            if 'g_r_squared' not in feature_df.columns:
                                feature_df['g_r_squared'] = feature_df['g_r'] ** 2
                            
                            # Veriyi ölçeklendir
                            X_scaled = scaler.transform(feature_df)
                            
                            # Tahmin yap
                            start_time = time.time()
                            rf_probs = rf.predict_proba(X_scaled)
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
                else:
                    missing = [col for col in required_cols if col not in df.columns]
                    st.error(f"CSV dosyasında gerekli sütunlar eksik: {', '.join(missing)}")
            except Exception as e:
                st.error(f"CSV dosyası işlenirken hata oluştu: {str(e)}")
    
    # ---------------------------------------------------------
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
            sample = make_feature_vector(
                example['u'], 
                example['g'], 
                example['r'], 
                example['i'], 
                example['z']
            )
            
            with st.spinner("Sınıflandırma yapılıyor..."):
                pred_class, confidence, class_probs = predict(sample, rf, scaler, labels)
                
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
