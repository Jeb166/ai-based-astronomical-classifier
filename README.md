# AI-based Astronomical Classifier (Optimize Edilmiş Sürüm)

Bu proje, astronomik nesneleri (galaksiler, kuasarlar ve yıldızlar) sınıflandırmak ve yıldızların alt türlerini tahmin etmek için eğitilmiş derin öğrenme modellerini içerir. 

## Optimizasyon

Projede yapılan optimizasyonlar:

- **Sadece Standart Model Mimarisi**: Farklı modellerin performans karşılaştırması sonucunda, standart model mimarisi en iyi performansı (%64.61 doğruluk) gösterdiği için, yalnızca bu mimari kullanılacak şekilde kod optimize edilmiştir.

- **Gereksiz Dosyaların Temizlenmesi**: Kullanılmayan boş dosyalar (`bayesian_optimize_star_fixed.py` ve `fix_csv.py`) silinmiştir.

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

3. Optimizasyon sonrası oluşturulan standart model sürümünü çalıştırın:
```
python src/main.py
```

## Dosyalar ve Açıklamaları

- `astronomy_classifier_optimized.ipynb`: Google Colab için optimize edilmiş Jupyter notebook
- `src/main.py`: Standart modeli kullanan optimize edilmiş ana program
- `src/star_model.py`: Standart model mimarisi tanımı
- `src/bayesian_optimize_star.py`: Standart model için Bayesian optimizasyon kodu

## Model Performansı

Standart model, diğer mimarilere göre en iyi performansı göstermiştir:

- Standart model: ~64.61% doğruluk
- Hafif model: ~51.30% doğruluk
- Ayrılabilir model: ~24.32% doğruluk
- Ağaç modeli: ~21.44% doğruluk

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