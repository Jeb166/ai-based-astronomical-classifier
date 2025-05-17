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
        <div style="text-align: center; width: 100%;">
            <div id="aladin-lite-div" style="height: 500px; width: 100%; border: 1px solid #ccc; border-radius: 5px; position: relative;"></div>
            <div id="status-message" style="margin-top: 10px; padding: 5px; color: #555; font-style: italic; background-color: #f8f9fa; border-radius: 3px;">Harita yükleniyor...</div>
            <div id="debug-log" style="display: none; margin-top: 5px; font-size: 12px; height: 60px; overflow-y: auto; background-color: #f1f1f1; color: #666; padding: 5px; border-radius: 3px;"></div>
        </div>
        <script type="text/javascript">
            // Debugging yardımcı fonksiyonları
            function debugLog(message) {
                console.log(message);
                try {
                    var debugElem = document.getElementById('debug-log');
                    if (debugElem) {
                        var now = new Date();
                        var timestamp = now.getHours() + ':' + now.getMinutes() + ':' + now.getSeconds();
                        debugElem.innerHTML += '<div>[' + timestamp + '] ' + message + '</div>';
                        debugElem.scrollTop = debugElem.scrollHeight;
                    }
                } catch (e) {
                    console.error("Debug log hatası:", e);
                }
            }

            function updateStatus(message, isError) {
                try {
                    var statusElem = document.getElementById('status-message');
                    if (statusElem) {
                        statusElem.innerHTML = message;
                        if (isError) {
                            statusElem.style.color = '#d9534f';
                            statusElem.style.backgroundColor = '#f2dede';
                        } else {
                            statusElem.style.color = '#555';
                            statusElem.style.backgroundColor = '#f8f9fa';
                        }
                    }
                } catch (e) {
                    console.error("Status güncelleme hatası:", e);
                }
            }            // Aladin script'ini dinamik olarak yükle
            function loadAladinScript() {
                updateStatus("Aladin kütüphanesi yükleniyor...");
                debugLog("Aladin script yükleme başladı");
                
                // jQuery yükleme başarısız olursa denemek için CDN'ler
                var jQueryCDNs = [
                    'https://code.jquery.com/jquery-3.6.0.min.js',
                    'https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js',
                    'https://cdn.jsdelivr.net/npm/jquery@3.6.0/dist/jquery.min.js'
                ];
                
                // Aladin yükleme başarısız olursa denemek için CDN'ler
                var aladinCDNs = [
                    'https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.min.js',
                    'https://cdn.jsdelivr.net/npm/aladin-lite@3.2.0/dist/aladin.min.js' // Alternatif CDN
                ];
                
                // Yükleme için son tarih - 20 saniye
                var loadingTimeout = setTimeout(function() {
                    if (typeof A === 'undefined') {
                        debugLog("Aladin yükleme zaman aşımı - 20 saniye geçti");
                        updateStatus("Aladin haritası yüklenemedi! Yükleme zaman aşımına uğradı.", true);
                        document.getElementById('debug-log').style.display = 'block';
                    }
                }, 20000);
                
                // Başarısız jQuery yüklemesi durumunda alternatif CDN'leri dene
                function tryLoadjQuery(index) {
                    if (index >= jQueryCDNs.length) {
                        debugLog("Tüm jQuery CDN'leri başarısız oldu");
                        updateStatus("jQuery kütüphanesi yüklenemedi! Lütfen internet bağlantınızı kontrol edin ve sayfayı yenileyin.", true);
                        document.getElementById('debug-log').style.display = 'block';
                        return;
                    }
                    
                    var jquerySrc = jQueryCDNs[index];
                    debugLog("jQuery yükleme deneniyor: " + jquerySrc);
                    
                    var jqueryScript = document.createElement('script');
                    jqueryScript.src = jquerySrc;
                    jqueryScript.type = 'text/javascript';
                    
                    jqueryScript.onload = function() {
                        debugLog("jQuery başarıyla yüklendi: " + jquerySrc);
                        loadAladinCSS();
                    };
                    
                    jqueryScript.onerror = function() {
                        debugLog("jQuery yükleme hatası: " + jquerySrc);
                        // Sonraki CDN'i dene
                        tryLoadjQuery(index + 1);
                    };
                    
                    document.head.appendChild(jqueryScript);
                }
                
                // Aladin CSS yükle
                function loadAladinCSS() {
                    try {
                        debugLog("Aladin CSS yükleniyor");
                        var alamdinCSS = document.createElement('link');
                        alamdinCSS.href = 'https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.min.css';
                        alamdinCSS.rel = 'stylesheet';
                        document.head.appendChild(alamdinCSS);
                        
                        // CSS yüklendikten sonra Aladin JS'yi yükle
                        tryLoadAladin(0);
                    } catch (e) {
                        debugLog("Aladin CSS yükleme hatası: " + e.message);
                        // Doğrudan Aladin JS'yi yüklemeye devam et
                        tryLoadAladin(0);
                    }
                }
                
                // Başarısız Aladin yüklemesi durumunda alternatif CDN'leri dene
                function tryLoadAladin(index) {
                    if (index >= aladinCDNs.length) {
                        debugLog("Tüm Aladin CDN'leri başarısız oldu");
                        updateStatus("Aladin kütüphanesi yüklenemedi! Lütfen internet bağlantınızı kontrol edin ve sayfayı yenileyin.", true);
                        document.getElementById('debug-log').style.display = 'block';
                        return;
                    }
                    
                    var aladinSrc = aladinCDNs[index];
                    debugLog("Aladin yükleme deneniyor: " + aladinSrc);
                    
                    var aladinScript = document.createElement('script');
                    aladinScript.src = aladinSrc;
                    aladinScript.type = 'text/javascript';
                    
                    aladinScript.onload = function() {
                        clearTimeout(loadingTimeout); // Zaman aşımını iptal et
                        debugLog("Aladin kütüphanesi başarıyla yüklendi: " + aladinSrc);
                        updateStatus("Aladin kütüphanesi yüklendi, harita hazırlanıyor...");
                        
                        // Aladin objesi tanımlı mı kontrol et
                        if (typeof A !== 'undefined') {
                            debugLog("Aladin global objesi (A) başarıyla tanımlandı");
                            setTimeout(initAladin, 500);
                        } else {
                            debugLog("Aladin global objesi (A) tanımlı değil");
                            // Aladin global nesnesinin tanımlanması için biraz daha bekle
                            var waitForA = setInterval(function() {
                                if (typeof A !== 'undefined') {
                                    clearInterval(waitForA);
                                    debugLog("Aladin global objesi (A) hazır");
                                    initAladin();
                                }
                            }, 100);
                            
                            // 5 saniye sonra hala tanımlı değilse, hataya geç
                            setTimeout(function() {
                                if (typeof A === 'undefined') {
                                    clearInterval(waitForA);
                                    debugLog("Aladin global objesi (A) tanımlanamadı");
                                    updateStatus("Aladin haritası başlatılamadı! Tarayıcı bileşeni uyumsuz olabilir.", true);
                                    document.getElementById('debug-log').style.display = 'block';
                                }
                            }, 5000);
                        }
                    };
                    
                    aladinScript.onerror = function() {
                        debugLog("Aladin yükleme hatası: " + aladinSrc);
                        // Sonraki CDN'i dene
                        tryLoadAladin(index + 1);
                    };
                    
                    document.head.appendChild(aladinScript);
                }
                
                // jQuery yüklemeyi başlat
                tryLoadjQuery(0);
            }
            }
              function initAladin() {
                try {
                    debugLog("Aladin haritası başlatılıyor");
                    updateStatus("Harita hazırlanıyor...");
                    
                    if (typeof A === 'undefined') {
                        debugLog("Aladin objesi tanımsız, yükleme başarısız");
                        updateStatus("Aladin haritası başlatılamadı! A objesi bulunamadı. Sayfayı yenileyin.", true);
                        document.getElementById('debug-log').style.display = 'block';
                        return;
                    }
                    
                    // Aladin konteynerinin hazır olduğundan emin ol
                    var divElement = document.getElementById('aladin-lite-div');
                    if (!divElement) {
                        debugLog("Aladin konteyneri bulunamadı");
                        updateStatus("Harita konteyneri bulunamadı!", true);
                        return;
                    }
                    
                    // Halihazırda başlatılmış bir Aladin örneği var mı kontrol et
                    if (window.aladinInstance) {
                        debugLog("Önceki Aladin örneği tespit edildi, temizleniyor");
                        try {
                            // Konteyneri temizle
                            divElement.innerHTML = '';
                        } catch (e) {
                            debugLog("Konteyner temizleme hatası: " + e.message);
                        }
                    }
                    
                    // Aladin nesnesini başlat
                    debugLog("A.aladin() çağrılıyor");
                    try {
                        var startTime = new Date().getTime();
                        var aladin = A.aladin('#aladin-lite-div', {
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
                            fullScreen: false,
                            showContextMenu: true,
                            log: function(message) {
                                debugLog("Aladin Log: " + message);
                            }
                        });
                        
                        var endTime = new Date().getTime();
                        debugLog("Aladin haritası başlatıldı (" + (endTime - startTime) + "ms)");
                        
                        // Global olarak sakla (yeniden başlatma/temizleme için)
                        window.aladinInstance = aladin;
                        
                        // SDSS DR18 katmanını ekle
                        debugLog("SDSS katmanı ekleniyor");
                        try {
                            var sdssLayer = A.imageLayer(
                                'SDSS DR18 Color',
                                'https://dr18.sdss.org/sas/dr18/eboss/photoObj/frames/color/', 
                                {survey: 'SDSS'}
                            );
                            aladin.addLayer(sdssLayer);
                            debugLog("SDSS katmanı eklendi");
                        } catch (e) {
                            debugLog("SDSS katmanı ekleme hatası: " + e.message);
                            // Ana harita çalışmaya devam edebilir, bu yüzden hatayı yut
                        }
                        
                        // Tıklamayı dinle
                        aladin.on('click', function(object) {
                            if (!object || typeof object.ra === 'undefined' || typeof object.dec === 'undefined') {
                                debugLog("Haritada geçersiz koordinat tıklaması");
                                return;
                            }
                            
                            const ra = object.ra.toFixed(6);
                            const dec = object.dec.toFixed(6);
                            debugLog("Haritada tıklama: RA=" + ra + ", Dec=" + dec);
                            updateStatus('Seçilen koordinatlar: RA=' + ra + '°, Dec=' + dec + '°');
                            
                            // Streamlit'e mesaj gönder
                            try {
                                parent.postMessage({type: 'aladin', ra: ra, dec: dec}, '*');
                                debugLog("Koordinat mesajı gönderildi");
                            } catch (e) {
                                debugLog("Koordinat mesaj hatası: " + e.message);
                            }
                        });
                        
                        // Harita yükleme olayını dinle
                        aladin.on('ready', function() {
                            debugLog("Aladin haritası tam olarak hazır");
                            updateStatus('Harita hazır. Herhangi bir noktaya tıklayarak koordinatları seçebilirsiniz.');
                        });
                        
                        // 5 saniye sonra harita hala yüklenmedi ise yükleme hatası olabilir
                        setTimeout(function() {
                            if (aladin && !aladin.isReady) {
                                debugLog("Harita 5 saniye içinde hazır olmadı, olası bir yükleme sorunu");
                                // Uyarı göster ama hataya geçmeden
                                updateStatus('Harita yükleniyor, lütfen bekleyin (bu biraz zaman alabilir)...');
                            }
                        }, 5000);
                        
                    } catch (e) {
                        debugLog("A.aladin() çağrısı hatası: " + e.message);
                        updateStatus('Harita başlatılırken hata oluştu: ' + e.message, true);
                        document.getElementById('debug-log').style.display = 'block';
                        
                        // Aladin.js'yi yeniden yüklemeyi dene
                        setTimeout(function() {
                            debugLog("Harita başlatma hatası sonrası yeniden deneme yapılıyor");
                            // Tüm script etiketlerini temizle ve yeniden yükle
                            var scripts = document.getElementsByTagName('script');
                            for (var i = 0; i < scripts.length; i++) {
                                if (scripts[i].src && scripts[i].src.includes('aladin')) {
                                    scripts[i].parentNode.removeChild(scripts[i]);
                                    i--; // Dizin azaldı
                                }
                            }
                            // Yeniden yüklemeyi dene
                            loadAladinScript();
                        }, 3000);
                    }
                    
                } catch (e) {
                    debugLog("Harita oluşturma hatası: " + e.message);
                    updateStatus('Harita başlatılırken hata oluştu: ' + e.message, true);
                    console.error('Aladin yükleme hatası:', e);
                    // Hata durumunda hata ayıklama menüsünü göster
                    document.getElementById('debug-log').style.display = 'block';
                    
                    // Tarayıcı uyumluluk kontrolü
                    debugLog("Tarayıcı uyumluluk kontrolü yapılıyor");
                    var isChrome = /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
                    var isFirefox = /Firefox/.test(navigator.userAgent);
                    var isEdge = /Edg/.test(navigator.userAgent);
                    var isSafari = /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent);
                    
                    var browserInfo = "Tarayıcı: ";
                    if (isChrome) browserInfo += "Chrome";
                    else if (isFirefox) browserInfo += "Firefox";
                    else if (isEdge) browserInfo += "Edge";
                    else if (isSafari) browserInfo += "Safari";
                    else browserInfo += "Bilinmeyen";
                    
                    browserInfo += " | Kullanıcı Aracısı: " + navigator.userAgent;
                    debugLog(browserInfo);
                }
            }
            
            // Scriptin yüklenmesini başlat
            debugLog("Sayfa yüklendi, Aladin yükleme başlatılıyor");
            // DOM yüklendikten sonra Aladin'i yükle
            if (document.readyState === "loading") {
                document.addEventListener("DOMContentLoaded", loadAladinScript);
            } else {
                loadAladinScript();
            }
        </script>        <style>
            .aladin-box {
                z-index: 1000 !important;
            }
            .aladin-layersControl-container {
                max-height: 200px;
                overflow-y: auto;
            }
            #aladin-lite-div:empty:before {
                content: "Harita yükleniyor...";
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: #888;
            }
            #status-message {
                transition: all 0.3s ease;
            }
            #retry-button {
                margin-top: 10px;
                padding: 5px 10px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                display: none;
            }
            #retry-button:hover {
                background-color: #0069d9;
            }
            .loading-animation {
                display: inline-block;
                width: 100%;
                height: 80px;
                background: url('https://cdn.jsdelivr.net/npm/svg-spinners@2.0.0/dist/simple-ring.svg') center center no-repeat;
                background-size: 40px;
                margin-top: 20px;
            }
        </style>
        <div id="aladin-fallback" style="display:none; text-align:center; padding: 20px; border: 1px solid #f8d7da; background-color: #f8d7da; color: #721c24; border-radius: 5px; margin-top: 10px;">
            <h3>Aladin Haritası Yüklenemedi</h3>
            <p>Aşağıdaki çözümleri deneyebilirsiniz:</p>
            <ol style="text-align: left; display: inline-block;">
                <li>Sayfayı yenileyin (F5)</li>
                <li>Farklı bir tarayıcı kullanın (Chrome, Firefox veya Edge önerilir)</li>
                <li>Tarayıcınızın JavaScript ayarlarını kontrol edin</li>
                <li>Güvenlik duvarınızın <code>aladin.u-strasbg.fr</code> adresine erişime izin verdiğinden emin olun</li>
                <li>Ağ bağlantınızı kontrol edin</li>
            </ol>
            <p>Alternatif olarak, aşağıdaki form alanlarını doğrudan doldurabilirsiniz.</p>
            <button id="retry-aladin" onclick="window.location.reload();">Sayfayı Yenile</button>
        </div>
        <script>
            // 10 saniye sonra hala Aladin yüklenmediyse, yedek mesajı göster
            setTimeout(function() {
                if (typeof A === 'undefined' || document.getElementById('status-message').textContent.includes('hata')) {
                    document.getElementById('aladin-fallback').style.display = 'block';
                    document.getElementById('retry-button').style.display = 'inline-block';
                }
            }, 10000);
        </script>
        """# JavaScript ile Streamlit arasında iletişimi sağlayan kod
        js_code = """
        <script type="text/javascript">
            // Debugging yardımcı fonksiyonu
            function logToConsole(message) {
                console.log("[StreamlitJS] " + message);
                // Debug div'i varsa ona da ekle
                try {
                    var debugElem = document.getElementById('debug-log');
                    if (debugElem) {
                        var now = new Date();
                        var timestamp = now.getHours() + ':' + now.getMinutes() + ':' + now.getSeconds();
                        debugElem.innerHTML += '<div>[' + timestamp + '] ' + message + '</div>';
                        debugElem.scrollTop = debugElem.scrollHeight;
                    }
                } catch (e) {
                    console.error("Debug log hatası:", e);
                }
            }

            // Form alanlarını bulan gelişmiş algoritma
            function findFormInputs() {
                logToConsole("Form alanları aranıyor...");
                try {
                    // Streamlit uygulamasındaki tüm belge
                    var doc = window.parent.document;
                    
                    // Önce Streamlit uygulamasının ana içerik bölümünü bul
                    var mainArea = doc.querySelector('.main');
                    if (!mainArea) {
                        logToConsole("Ana içerik alanı bulunamadı");
                        return null;
                    }
                    
                    // Tüm number inputları seç
                    var numberInputs = mainArea.querySelectorAll('input[type="number"]');
                    if (!numberInputs || numberInputs.length < 2) {
                        logToConsole("Yeterli sayıda input alanı bulunamadı, bulunan: " + (numberInputs ? numberInputs.length : 0));
                        return null;
                    }
                    
                    logToConsole("Toplam " + numberInputs.length + " input alanı bulundu");
                    
                    // RA ve Dec inputlarını bul
                    var raInput = null;
                    var decInput = null;
                    
                    // Her input için ebeveyn element zincirinde etiketlere bak
                    for (var i = 0; i < numberInputs.length; i++) {
                        // Input'un parent elementlerini kontrol et
                        var currentElement = numberInputs[i];
                        var found = false;
                        var depth = 0;
                        var maxDepth = 5; // Maksimum yukarı çıkılacak ebeveyn sayısı
                        
                        while (currentElement && depth < maxDepth && !found) {
                            // Eğer etiket (label) element varsa kapsayıcıda, onu kontrol et
                            var labelElements = currentElement.parentElement.querySelectorAll('label, .st-emotion-cache-1vzeuhh');
                            
                            for (var j = 0; j < labelElements.length; j++) {
                                var labelText = labelElements[j].textContent.toLowerCase();
                                
                                // Text içeriğini kontrol et
                                if (labelText.includes('sağ açıklık') || labelText.includes('ra')) {
                                    raInput = numberInputs[i];
                                    found = true;
                                    logToConsole("RA input alanı bulundu: " + labelText);
                                    break;
                                } else if (labelText.includes('dik açıklık') || labelText.includes('dec')) {
                                    decInput = numberInputs[i];
                                    found = true;
                                    logToConsole("Dec input alanı bulundu: " + labelText);
                                    break;
                                }
                            }
                            
                            // Bir seviye yukarı çık
                            currentElement = currentElement.parentElement;
                            depth++;
                        }
                    }
                    
                    // Sonuçları döndür
                    if (raInput && decInput) {
                        logToConsole("RA ve Dec alanları başarıyla bulundu");
                        return { ra: raInput, dec: decInput };
                    } else {
                        // İlk iki input'u varsayılan olarak kullan
                        if (numberInputs.length >= 2) {
                            logToConsole("Etiketler bulunamadı, varsayılan olarak ilk iki input kullanılıyor");
                            return { ra: numberInputs[0], dec: numberInputs[1] };
                        }
                        logToConsole("RA veya Dec alanı bulunamadı");
                        return null;
                    }
                } catch (err) {
                    logToConsole("Form alanları aranırken hata: " + err.message);
                    console.error("Form alanları hatası:", err);
                    return null;
                }
            }
            
            // Aladin'den gelen koordinatları form alanlarına yerleştir
            function updateCoordinates(ra, dec) {
                logToConsole("Koordinatları güncelleme işlemi başlatıldı: RA=" + ra + ", Dec=" + dec);
                
                // 3 deneme yap (100ms aralarla)
                var attempts = 0;
                var maxAttempts = 5;
                
                function tryUpdate() {
                    attempts++;
                    logToConsole("Deneme #" + attempts);
                    
                    var inputs = findFormInputs();
                    if (inputs) {
                        try {
                            // Değerleri güncelle
                            inputs.ra.value = ra;
                            inputs.dec.value = dec;
                            
                            // Değişikliği tetikle (Streamlit için gerekli)
                            inputs.ra.dispatchEvent(new Event('input', { bubbles: true }));
                            inputs.dec.dispatchEvent(new Event('input', { bubbles: true }));
                            
                            logToConsole("Koordinatlar başarıyla güncellendi");
                            
                            // Eğer varsayılan bir statusMessage varsa, onu da güncelle
                            var statusMessage = document.getElementById('status-message');
                            if (statusMessage) {
                                statusMessage.innerHTML = 'Koordinatlar form alanlarına yerleştirildi: RA=' + ra + '°, Dec=' + dec + '°';
                            }
                            
                            return true;
                        } catch (err) {
                            logToConsole("Koordinat güncellerken hata: " + err.message);
                            console.error("Koordinat güncelleme hatası:", err);
                        }
                    }
                    
                    // Maksimum deneme sayısına ulaşılmadıysa tekrar dene
                    if (attempts < maxAttempts) {
                        setTimeout(tryUpdate, 200); // 200ms sonra tekrar dene
                    } else {
                        logToConsole("Form alanları " + maxAttempts + " denemeden sonra güncellenemedi");
                        
                        // Kullanıcıya bildir
                        var statusMessage = document.getElementById('status-message');
                        if (statusMessage) {
                            statusMessage.innerHTML = 'Seçilen koordinatlar (' + ra + '°, ' + dec + '°) form alanlarına yerleştirilemedi. Lütfen manuel girin.';
                            statusMessage.style.color = '#d9534f';
                        }
                    }
                }
                
                // İlk denemeyi başlat
                setTimeout(tryUpdate, 100);
            }
            
            // Koordinat mesajlarını dinle
            window.addEventListener('message', function(event) {
                if (event.data && event.data.type === 'aladin') {
                    const ra = event.data.ra;
                    const dec = event.data.dec;
                    
                    logToConsole('Koordinat mesajı alındı: RA=' + ra + ', Dec=' + dec);
                    
                    // Form alanlarını güncelle
                    updateCoordinates(ra, dec);
                }
            });
            
            // İletişimin aktif olduğunu göster
            logToConsole('JavaScript-Streamlit iletişimi başlatıldı (v1.2)');
        </script>
        """        # Aladin ve JavaScript'i ekle
        debug_mode = st.sidebar.checkbox("Hata Ayıklama Modunu Etkinleştir", value=False, help="Harita yüklenmiyorsa bu seçeneği etkinleştirin")
        
        # Hata ayıklama moduna göre JavaScript kodunu ayarla
        if debug_mode:
            # Debug modu etkinse görünür kıl
            aladin_html = aladin_html.replace('id="debug-log" style="display: none;', 'id="debug-log" style="display: block;')
        
        components.html(aladin_html + js_code, height=600 if debug_mode else 550)
        
        # Alternatif koordinat giriş seçenekleri için bilgilendirme
        st.info("""
        **Not:** Eğer harita yüklenme sorunu yaşıyorsanız, SDSS Navigator URL'sini kullanarak veya 
        aşağıdaki form alanlarına doğrudan koordinat girerek arama yapabilirsiniz.
        """)        # Kullanıcıya yönergeler
        with st.expander("Harita Kullanım Yönergeleri", expanded=True):
            st.markdown("""
            **Gökyüzü haritası nasıl kullanılır?**
            1. Harita yüklendikten sonra, herhangi bir noktaya tıklayarak o koordinatları seçebilirsiniz.
            2. Yakınlaştırmak/uzaklaştırmak için sağ üst köşedeki + ve - düğmelerini veya fare tekerleğini kullanabilirsiniz.
            3. Katman kontrolü ile farklı astronomik görüntü servislerini seçebilirsiniz.
            
            **Harita görünmüyorsa veya yüklenemiyorsa:**
            1. Sayfayı yenileyin (F5 tuşu)
            2. Tarayıcı önbelleğini temizleyin (Ctrl+Shift+Delete)
            3. Tarayıcınızın JavaScript'i etkinleştirdiğinden emin olun
            4. Farklı bir tarayıcı deneyin (Chrome, Firefox veya Edge önerilir)
            5. Tarayıcınızın içerik engelleme ayarlarını kontrol edin ve `aladin.u-strasbg.fr` adresinden içeriğe izin verin
            6. Güvenlik duvarınızın astronomik servislere erişime izin verdiğinden emin olun
            7. Ağ bağlantınızı kontrol edin - kararlı bir internet bağlantısı gereklidir
            
            **Not:** Eğer harita görünmüyorsa ve yukarıdaki çözümler işe yaramıyorsa, koordinatları doğrudan form alanlarına girebilir veya 
            SDSS Navigator URL'sini kullanabilirsiniz.
            """)
        
        # Harita hata kontrolü - ayrı bir expander olarak
        st.markdown("### Harita Sorun Giderme")
        st.warning("Harita yüklenme sorunlarını gidermek için aşağıdaki araçları kullanabilirsiniz")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Aladin Durumunu Kontrol Et"):
                st.code("""
                // Bu kodu tarayıcınızın konsol penceresinde (F12) çalıştırın:
                if (typeof A === 'undefined') {
                    console.log("Aladin kütüphanesi yüklenemedi!");
                } else {
                    console.log("Aladin kütüphanesi yüklendi ve hazır.");
                }
                
                // Aladin div'inin durumunu kontrol et
                var div = document.getElementById('aladin-lite-div');
                console.log("Aladin div durumu:", div ? "Bulundu" : "Bulunamadı");
                if (div) {
                    console.log("Div boyutları:", div.offsetWidth, "x", div.offsetHeight);
                    console.log("Div içeriği:", div.innerHTML.substr(0, 100) + "...");
                }
                """, language="javascript")
        
        with col2:
            if st.button("Haritayı Yeniden Yükle"):
                st.code("""
                // Bu kodu tarayıcınızın konsol penceresinde (F12) çalıştırın:
                try {
                    // Önceki Aladin örneğini temizle
                    var div = document.getElementById('aladin-lite-div');
                    if (div) {
                        div.innerHTML = '';
                        console.log("Aladin div'i temizlendi");
                    }
                    
                    // Tüm aladin scriptlerini temizle
                    var scripts = document.getElementsByTagName('script');
                    for (var i = 0; i < scripts.length; i++) {
                        if (scripts[i].src && scripts[i].src.includes('aladin')) {
                            scripts[i].parentNode.removeChild(scripts[i]);
                            console.log("Aladin script etiketi kaldırıldı");
                        }
                    }
                    
                    // Sayfayı yenile
                    window.location.reload();
                } catch (e) {
                    console.error("Hata:", e);
                }
                """, language="javascript")
        
        st.info("Eğer sorun devam ediyorsa, tarayıcınızın konsol çıktısını kontrol edin (F12 tuşu > Console sekmesi) ve hata mesajlarını inceleyin.")
        st.info("Bu sorunu çözmek için Streamlit uygulamasını yeniden başlatmayı da deneyebilirsiniz.")
        
        # Ağ bağlantı kontrolü
        if st.button("Ağ Bağlantısını Kontrol Et"):
            components.html("""
            <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                <h4>Ağ Bağlantı Durumu</h4>
                <div id="network-status">Kontrol ediliyor...</div>
                <script>
                    var statusDiv = document.getElementById('network-status');
                    
                    // Internet bağlantısını kontrol et
                    function checkConnection() {
                        statusDiv.innerHTML = "Kontrol ediliyor...";
                        
                        // Ping çeşitli servisler
                        var services = [
                            {name: "jQuery CDN", url: "https://code.jquery.com/jquery-3.6.0.min.js"},
                            {name: "Aladin CDN", url: "https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.min.js"},
                            {name: "SDSS", url: "https://dr18.sdss.org/"}
                        ];
                        
                        var results = "<ul>";
                        var completedChecks = 0;
                        
                        services.forEach(function(service) {
                            var startTime = new Date().getTime();
                            var xhr = new XMLHttpRequest();
                            xhr.open('HEAD', service.url, true);
                            
                            xhr.onload = function() {
                                var endTime = new Date().getTime();
                                var pingTime = endTime - startTime;
                                var status = (xhr.status >= 200 && xhr.status < 400) ? "✅ Erişilebilir" : "❌ Erişilemiyor";
                                
                                results += "<li>" + service.name + ": " + status + " (Yanıt süresi: " + pingTime + "ms)</li>";
                                
                                completedChecks++;
                                if (completedChecks === services.length) {
                                    results += "</ul>";
                                    statusDiv.innerHTML = results;
                                }
                            };
                            
                            xhr.onerror = function() {
                                results += "<li>" + service.name + ": ❌ Bağlantı hatası</li>";
                                
                                completedChecks++;
                                if (completedChecks === services.length) {
                                    results += "</ul>";
                                    statusDiv.innerHTML = results;
                                }
                            };
                            
                            xhr.send();
                        });
                    }
                    
                    checkConnection();
                </script>
            </div>
            """, height=250)
        
        
        
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
