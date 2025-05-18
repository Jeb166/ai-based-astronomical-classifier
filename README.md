# 🔭 AI-based Astronomical Classifier 

Bu proje, astronomik nesneleri (galaksiler, kuasarlar ve yıldızlar) sınıflandırmak ve yıldızların alt türlerini tahmin etmek için eğitilmiş makine öğrenmesi ve derin öğrenme modellerini içerir. SDSS (Sloan Digital Sky Survey) verileri üzerinde eğitilmiş modellerle %99'a varan doğrulukla temel sınıflandırma yapabilmektedir.

## 🚀 Özellikler ve Yenilikler

- **Yüksek Doğruluklu Temel Sınıflandırma**: Galaksi, kuasar ve yıldız sınıflandırmasında %99 doğruluk.
- **Optimize Random Forest Algoritması**: Yüksek hız ve doğruluk dengesini sağlayan geliştirilmiş sınıflandırma algoritması.
- **Streamlit Web Arayüzü**: Kullanıcı dostu web arayüzü ile gökyüzü nesnelerini anında analiz edebilme imkanı.
- **Otomatik Parametre Optimizasyonu**: Kullanıcının manuel konfigürasyon yapmasına gerek kalmadan optimum performans.
- **SDSS API Entegrasyonu**: Koordinatlarla gerçek zamanlı gökyüzü görüntüsü ve spektrum verisi erişimi.

## 🔧 Kurulum ve Çalıştırma

### Yerel Bilgisayarda Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/yourusername/ai-based-astronomical-classifier.git
cd ai-based-astronomical-classifier
```

2. Sanal ortam oluşturun (opsiyonel ama önerilir):
```bash
python -m venv venv
# Windows'da
venv\Scripts\activate
# Linux/Mac'de
# source venv/bin/activate
```

3. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

### Çalıştırma Seçenekleri

#### 1. Web Arayüzü İle Çalıştırma (Önerilen)
```bash
streamlit run src/streamlit.py
```
Bu komut, tarayıcınızda otomatik olarak Streamlit arayüzünü açacaktır.

#### 2. Komut Satırı Versiyonu
```bash
python src/main.py
```
Bu komut, temel Random Forest modelini eğitecek ve sonuçları konsolda gösterecektir.

## 📂 Proje Yapısı

```
ai-based-astronomical-classifier/
│
├── src/                  # Kaynak kod
│   ├── main.py           # Ana program (CLI versiyonu)
│   ├── streamlit.py      # Web arayüzü (Streamlit app)
│   ├── prediction.py     # Tahmin fonksiyonları
│   ├── prepare_data.py   # Veri hazırlama modülü
│   └── data_analysis.py  # Veri analiz araçları
│
├── data/                 # Veri dosyaları
│   ├── skyserver.csv     # Ana veri seti
│   └── skyserver_test_data.csv # Test veri seti
│
├── outputs/              # Model çıktıları ve grafikler
│   ├── rf_model.joblib   # Eğitilmiş Random Forest modeli
│   └── scaler.joblib     # Özellik ölçekleyici
│
├── backups/              # Eski model referansları
│
├── requirements.txt      # Gereksinimler
└── README.md             # Bu dokümantasyon
```

## 📊 Model Performansı

- **Random Forest Modeli**: Galaksi/Kuasar/Yıldız ayrımında ~99% doğruluk

## 🔧 Teknik Detaylar

Kullanılan temel özellikler:
- 5 band fotometrik magnitude (u, g, r, i, z)
- 4 renk indeksi (u-g, g-r, r-i, i-z)
- Redshift ve diğer spektroskopik ölçümler

Model mimarisi:
- Random Forest ile temel sınıflandırma (500 ağaç ve optimize edilmiş hiperparametreler)

## 📦 Gereksinimler

```
pandas >= 1.3.0
numpy >= 1.20.0
scikit-learn >= 1.0.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
joblib >= 1.1.0
streamlit >= 1.18.0
requests >= 2.28.0
pillow >= 9.0.0
plotly >= 5.10.0
astroquery >= 0.4.6
astropy >= 5.0.0
```

## 📝 Kullanım Örnekleri

### Python Kodunda Kullanım

```python
import joblib
import numpy as np

# Modeli ve scaler'ı yükle
model = joblib.load('outputs/rf_model.joblib')
scaler = joblib.load('outputs/scaler.joblib')

# Veri ön işleme
sample = np.array([[18.5, 17.2, 16.8, 16.5, 16.3, 0.2, 0.3, 0.1]])  # u,g,r,i,z,redshift,...
sample_scaled = scaler.transform(sample)

# Tahmin
prediction = model.predict(sample_scaled)
probabilities = model.predict_proba(sample_scaled)

print(f"Tahmin edilen sınıf: {prediction}")
print(f"Sınıf olasılıkları: {probabilities}")
```

### Web Arayüzünde Kullanım

1. Streamlit uygulamasını başlatın: `streamlit run src/streamlit.py`
2. Koordinat veya özellik değerleri girin
3. "Sınıflandır" butonuna tıklayın

## 📃 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakın.

## 📧 İletişim

Sorularınız ve önerileriniz için: emre@example.com

---
*Son güncelleme: Mayıs 2025*