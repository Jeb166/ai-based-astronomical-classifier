# 🔭 AI-based Astronomical Classifier

Bu proje, astronomik nesneleri (galaksiler, kuasarlar ve yıldızlar) sınıflandırmak için geliştirilmiş bir makine öğrenmesi modelini içermektedir. SDSS (Sloan Digital Sky Survey) verileri üzerinde eğitilmiş model, temel sınıflandırma görevlerinde yüksek doğruluk sağlamaktadır.

## Özellikler

- **Temel Sınıflandırma**: Galaksi, kuasar ve yıldız sınıflandırmasında yüksek doğruluk.
- **Random Forest Algoritması**: Dengeli bir hız ve doğruluk sağlayan optimize edilmiş bir sınıflandırma algoritması.
- **Streamlit Web Arayüzü**: Kullanıcıların sınıflandırma işlemlerini kolayca gerçekleştirebileceği bir web arayüzü.

## Kurulum ve Çalıştırma

### Yerel Bilgisayarda Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/yourusername/ai-based-astronomical-classifier.git
cd ai-based-astronomical-classifier
```

2. Sanal ortam oluşturun (opsiyonel):
```bash
python -m venv venv
# Windows'da
venv\Scripts\activate
# Linux/Mac'de
source venv/bin/activate
```

3. Gereksinimleri yükleyin:
```bash
pip install -r requirements.txt
```

### Çalıştırma Adımları

#### 1. Modeli Eğitme

Modeli kullanmadan önce eğitmeniz gerekmektedir. Bunun için aşağıdaki komutu çalıştırın:
```bash
python src/main.py
```
Bu işlem, modeli eğitecek ve gerekli dosyaları `outputs/` klasörüne kaydedecektir.

#### 2. Web Arayüzü İle Kullanım

Model eğitildikten sonra Streamlit arayüzünü başlatabilirsiniz:
```bash
streamlit run src/streamlit.py
```
Bu komut, tarayıcınızda Streamlit arayüzünü açacaktır.

## Proje Yapısı

```
ai-based-astronomical-classifier/
│
├── src/                  # Kaynak kod
│   ├── main.py           # Ana program (model eğitimi)
│   ├── streamlit.py      # Web arayüzü
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

## Model Performansı

- **Random Forest Modeli**: Galaksi/Kuasar/Yıldız ayrımında yüksek doğruluk.

## Teknik Detaylar

Kullanılan temel özellikler:
- 5 band fotometrik magnitude (u, g, r, i, z)
- 4 renk indeksi (u-g, g-r, r-i, i-z)
- Redshift ve diğer spektroskopik ölçümler

Model mimarisi:
- Random Forest ile temel sınıflandırma (500 ağaç ve optimize edilmiş hiperparametreler)

## Gereksinimler

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

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakın.

## İletişim

Sorularınız için: emre@example.com