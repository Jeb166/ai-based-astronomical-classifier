import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
from PIL import Image
from io import BytesIO
import streamlit.components.v1 as components
import urllib.parse as ul
import requests

# Model işlevlerini ve tahmin işlevlerini içe aktar
from prediction import (
    load_models, predict_optimized, get_sdss_image, get_sdss_spectrum, 
    get_sdss_photometry, extract_features_from_photometry, make_feature_vector
)

# En yakın SDSS objesini bulmak için yardımcı fonksiyon
import requests
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

st.title("AI Tabanlı Astronomik Gök Cismi Sınıflandırıcı")
st.markdown("""
Bu uygulama, derin öğrenme ve makine öğrenmesi yöntemleri kullanarak astronomik gök cisimlerini 
sınıflandırır. SDSS verilerini kullanarak galaksi, kuasar ve yıldız tespiti yapabilirsiniz.
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
    ["Koordinat ile Arama", "CSV Dosyası Yükleme", "Örnek Veriler"]
)

# Modellleri yükle
dnn, rf, scaler, labels, best_w = load_models()

if dnn is not None and rf is not None:
    st.sidebar.success("Modeller başarıyla yüklendi! 🚀")
    
    # Koordinat ile arama
    if input_method == "Koordinat ile Arama":
        st.subheader("Koordinat ile Gök Cismi Ara")
        
        # Aladin Lite harita görüntüleyici ekleme
        st.markdown("### Etkileşimli Gökyüzü Haritası")
        st.markdown("Aşağıdaki haritada herhangi bir noktaya tıklayarak o koordinatları otomatik olarak seçebilirsiniz.")
        
        import streamlit.components.v1 as components
        
        aladin_html = """
        <div id="aladin-lite-div" style="height:500px;"></div>
        <script src="https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.min.js"></script>
        <link rel="stylesheet" href="https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.min.css">
        <script>
            let aladin;
            
            function initAladin() {
                aladin = A.aladin('#aladin-lite-div', {
                    survey: "P/DSS2/color",
                    fov: 0.5,
                    target: "180 0",
                    cooFrame: "J2000",
                    reticleSize: 22,
                    showReticle: true,
                    showZoomControl: true,
                    showFullscreenControl: true,
                    showLayersControl: true,
                    showGotoControl: true,
                    showFrame: true,
                    fullScreen: false
                });
                
                // SDSS DR18 katmanını ekle
                aladin.createImageSurvey('SDSS DR18 Color', 'SDSS DR18 Color', 
                    'https://dr18.sdss.org/sas/dr18/eboss/photoObj/frames/color/', 
                    'SDSSg', 'SDSSi', '2mass');
                    
                // Tıklamayı dinle
                aladin.on('click', function(object) {
                    const ra = object.ra.toFixed(6);
                    const dec = object.dec.toFixed(6);
                    // Streamlit'e mesaj gönder
                    parent.postMessage({type: 'aladin', ra: ra, dec: dec}, '*');
                });
            }
            
            document.addEventListener('DOMContentLoaded', function() {
                initAladin();
            });
        </script>
        <style>
            .aladin-box {
                z-index: 1000 !important;
            }
        </style>
        """
        
        # JavaScript ile Streamlit arasında iletişimi sağlayan kod
        js_code = """
        <script>
            window.addEventListener('message', function(event) {
                if (event.data.type === 'aladin') {
                    const ra = event.data.ra;
                    const dec = event.data.dec;
                    
                    // Streamlit uygulamasına veriyi gönder
                    const raInput = window.parent.document.querySelector('input[data-testid="stNumberInput"][aria-label="Sağ açıklık (RA, derece)"]');
                    const decInput = window.parent.document.querySelector('input[data-testid="stNumberInput"][aria-label="Dik açıklık (Dec, derece)"]');
                    
                    if (raInput && decInput) {
                        // Değerleri güncelle
                        raInput.value = ra;
                        decInput.value = dec;
                        
                        // Değişikliği tetikle
                        raInput.dispatchEvent(new Event('input', { bubbles: true }));
                        decInput.dispatchEvent(new Event('input', { bubbles: true }));
                        
                        // Kullanıcıya bildir
                        console.log('Koordinatlar güncellendi: RA=' + ra + '°, Dec=' + dec + '°');
                    }
                }
            });
        </script>
        """        # Aladin ve JavaScript'i ekle
        components.html(aladin_html + js_code, height=520)
        
        # Kullanıcıya bilgilendir
        st.info("Haritada bir noktaya tıklandığında, koordinatlar otomatik olarak form alanlarına doldurulacaktır.")
        
        # SDSS Navigator URL'inden koordinat çıkarma
        import urllib.parse as ul
        url = st.text_input("SDSS Navi URL'sini yapıştır (örn: https://skyserver.sdss.org/dr18/VisualTools/navi?ra=146.47818&dec=41.73088...)")
        if url and "ra=" in url and "dec=" in url:
            qs = ul.parse_qs(ul.urlparse(url).query)
            ra_url = float(qs.get("ra", [0])[0])            
            dec_url = float(qs.get("dec", [0])[0])
            # Session state kullanarak giriş alanlarını güncelle
            st.session_state["ra"] = ra_url
            st.session_state["dec"] = dec_url
            st.success(f"Koordinatlar yakalandı: RA={ra_url:.4f}°, Dec={dec_url:.4f}°")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ra = st.number_input("Sağ açıklık (RA, derece)", min_value=0.0, max_value=360.0, value=180.0, key="ra")
        
        with col2:
            dec = st.number_input("Dik açıklık (Dec, derece)", min_value=-90.0, max_value=90.0, value=0.0, key="dec")
            
        # Arama seçenekleri
        col1, col2 = st.columns(2)
        with col1:
            search_option = st.radio(
                "Arama seçeneği:",
                ["Direkt Koordinat", "En Yakın SDSS Objesini Bul"],
                index=0,
                help="Direkt koordinat: Tam olarak girilen RA/Dec kullanılır. En yakın: Girilen koordinata en yakın SDSS objesi bulunur."
            )
        with col2:
            radius = st.number_input(
                "Arama yarıçapı (derece)",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                format="%.3f",
                help="En yakın obje araması için kullanılacak yarıçap (derece cinsinden). 0.01° yaklaşık 36 açı saniyesine eşittir."
            )
        
        if st.button("Ara ve Sınıflandır", type="primary"):
            # İşlem başladı mesajı
            with st.spinner("SDSS'den veri alınıyor ve analiz ediliyor..."):
                # En yakın objeyi bulma seçeneği etkinse
                if search_option == "En Yakın SDSS Objesini Bul":
                    nearest_objs = query_nearest_obj(ra, dec, radius)
                    
                    if nearest_objs is not None and not nearest_objs.empty:
                        # İlk objeyi seç (en yakın)
                        nearest_obj = nearest_objs.iloc[0]
                        # Koordinatları güncelle
                        ra = nearest_obj.get('ra', ra)
                        dec = nearest_obj.get('dec', dec)
                        st.success(f"En yakın obje bulundu: RA={ra:.4f}°, Dec={dec:.4f}°")
                    else:
                        st.warning(f"Belirtilen koordinatlarda ({ra:.4f}°, {dec:.4f}°) ve {radius}° yarıçapında hiçbir SDSS objesi bulunamadı.")
                
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
                                # Olasılıkları normalize et, 0-100 arasında göster
                                normalized_probs = all_probs[0] / np.sum(all_probs[0]) * 100
                                probs_df = pd.DataFrame({
                                    'Sınıf': labels,
                                    'Olasılık (%)': normalized_probs
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
                            # Her satır için ayrı ayrı tahmin yapalım
                            # Böylece her bir gök cisminin nasıl değerlendirildiğini görebiliriz
                            predictions_list = []
                            probabilities_list = []
                            all_probs_list = []
                            # Gerçek sınıf bilgisini sakla (varsa)
                            has_true_class = 'class' in df.columns
                            
                            for idx, row in df.iterrows():
                                st.write(f"------- {idx+1}. satır analiz ediliyor -------")
                                if has_true_class:
                                    st.write(f"Gerçek sınıf: {row['class']}")
                                    
                                # Tek satırlık feature vektörü oluştur
                                feature = make_feature_vector(row['u'], row['g'], row['r'], row['i'], row['z'])
                                
                                # Model tahminlerini ayrı ayrı göster (RF ve DNN ham tahminleri)
                                # StandardScaler ile verileri ölçeklendir
                                X = scaler.transform(feature)
                                
                                # Her iki modelden ham tahminleri al (ensemble öncesi)
                                dnn_probs_raw = dnn.predict(X, verbose=0)
                                rf_probs_raw = rf.predict_proba(X)
                                
                                st.write("**Ham model tahminleri:**")
                                st.write(f"- RF modeli: GALAXY={rf_probs_raw[0][0]:.4f}, QSO={rf_probs_raw[0][1]:.4f}, STAR={rf_probs_raw[0][2]:.4f}")
                                st.write(f"- DNN modeli: GALAXY={dnn_probs_raw[0][0]:.4f}, QSO={dnn_probs_raw[0][1]:.4f}, STAR={dnn_probs_raw[0][2]:.4f}")
                                
                                # Bu tek örnek için optimize edilmiş tahmin yap
                                prediction, probability, all_prob = predict_optimized(feature, dnn, rf, scaler, labels)
                                
                                # Sonuçları listelere ekle
                                predictions_list.append(prediction[0])
                                probabilities_list.append(probability[0])
                                all_probs_list.append(all_prob[0])
                                
                                # Tahmin sonuçlarını göster
                                st.write("**Tahmin sonucu:**", prediction[0])
                                st.write(f"**Güven düzeyi:** {probability[0]*100:.2f}%")
                                st.write("**Son olasılıklar:**")
                                st.write(f"- GALAXY={all_prob[0][0]:.4f}, QSO={all_prob[0][1]:.4f}, STAR={all_prob[0][2]:.4f}")
                                st.write("---")
                            
                            # Tüm tahminler tamamlandıktan sonra DataFrame'e ekle
                            if len(predictions_list) == len(df):
                                # Tahmin sonuçlarını DataFrame'e ekle
                                df['predicted_class'] = predictions_list
                                df['confidence'] = [p * 100 for p in probabilities_list]
                                
                                # Olasılık sütunlarını ekle (normalize edilmiş şekilde)
                                # Önce tüm olasılık sütunlarını oluşturalım (boş olarak)
                                for label in labels:
                                    df[f'prob_{label}'] = 0.0  # Varsayılan değer olarak 0.0 atayalım
                                
                                # Sonra her satır için olasılık değerlerini ekleyelim
                                for idx, prob_vector in enumerate(all_probs_list):
                                    # Her satır için olasılıkları toplamlarına bölerek normalize et
                                    normalized_probs = prob_vector / np.sum(prob_vector) * 100
                                    for i, label in enumerate(labels):
                                        df.at[idx, f'prob_{label}'] = normalized_probs[i]
                                
                                # Sonuçları göster
                                st.subheader("Sınıflandırma Sonuçları")
                                st.dataframe(df)
                                
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
            except Exception as e:
                st.error(f"CSV dosyası işlenirken hata oluştu: {str(e)}")
    
    # Örnek veriler
    elif input_method == "Örnek Veriler":
        st.subheader("Örnek Verilerle Tanıtım")
        # Örnek gök cisimleri - SDSS veri dağılımlarına göre optimize edilmiş
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
                                # Olasılıkları normalize et, 0-100 arasında göster
                                normalized_probs = all_probs[0] / np.sum(all_probs[0]) * 100
                                probs_df = pd.DataFrame({
                                    'Sınıf': labels,
                                    'Olasılık (%)': normalized_probs
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
