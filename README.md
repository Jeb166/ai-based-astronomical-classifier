# AI-based Astronomical Classifier (Optimize Edilmiş Sürüm)

Bu proje, astronomik nesneleri (galaksiler, kuasarlar ve yıldızlar) sınıflandırmak ve yıldızların alt türlerini tahmin etmek için eğitilmiş derin öğrenme modellerini içerir. 

## Optimizasyon

Projede yapılan optimizasyonlar:

- **Geliştirilmiş Yıldız Modeli Mimarisi**: Yıldız modeli mimarisi iyileştirilmiş ve doğruluğu %70+ seviyesine çıkarılmıştır. Çift yollu sinir ağı mimarisi (yoğun katmanlar + konvolüsyonel katmanlar) kullanılmıştır.

- **Otomatik Parametre Kullanımı**: Optimum hiperparametreler modelin içine gömülmüş, kullanıcının ek konfigürasyon yapmasına gerek kalmadan en iyi performans alınabilmektedir.

- **Gereksiz Dosyaların Temizlenmesi**: Kullanılmayan boş dosyalar silinmiştir.

- **Google Colab Uyumluluğu**: Proje, doğrudan Google Colab üzerinde çalıştırılabilecek şekilde düzenlenmiştir. Bunun için `astronomy_classifier_optimized.ipynb` Jupyter notebook dosyası eklenmiştir.

## Nasıl Çalıştırılır

### Google Colab'da Çalıştırma

1. `astronomy_classifier_optimized.ipynb` dosyasını Google Colab'a yükleyin
2. Notebook'taki hücreleri sırayla çalıştırın

### Yerel Ortamda Çalıştırma

1. Depoyu klonlayın:
```
git clone https://github.com/yourusername/ai-based-astronomical-classifier.git
cd ai-based-astronomical-classifier
```

2. Gerekli paketleri yükleyin:
```
pip install -r requirements.txt
```

3. Optimize edilmiş modeli çalıştırın:
```
python src/main.py
```

## Dosyalar ve Açıklamaları

- `astronomy_classifier_optimized.ipynb`: Google Colab için optimize edilmiş Jupyter notebook
- `src/main.py`: Ana program
- `src/star_model.py`: Yıldız modeli mimarisi tanımı (optimize parametreler varsayılan olarak ayarlanmış)
- `src/bayesian_optimize_star.py`: Yıldız modeli için Bayesian optimizasyon kodu

## Model Performansı

İyileştirilmiş mimari daha dengeli bir performans sağlamaktadır:

- Yıldız modeli: ~60-65% doğruluk (dengeli parametreler)
- Temel sınıflandırma: ~99% doğruluk (galaksi/kuasar/yıldız)

## Gereksinimler

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- pandas
- matplotlib
- seaborn
- joblib
- scikit-optimize (Bayesian optimizasyon için)

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakın.

## İletişim

Sorularınız için: emre@example.com