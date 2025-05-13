# Gerekli paketleri yükleyin ve projeyi çalıştırın
# Google Colab ortamında en yüksek performans gösteren standart modeli kullanan sürüm

# Paketi yükle
!pip install -q scikit-learn pandas matplotlib tensorflow joblib seaborn scikit-optimize

# GitHub'dan optimize edilmiş sürümü klonla
!git clone https://github.com/yourusername/ai-based-astronomical-classifier.git
%cd ai-based-astronomical-classifier

# Gerekirse ortamı kur
!pip install -r requirements.txt

# Optimize edilmiş versiyonu çalıştır
# Standard modeli kullanarak eğitim, varsayılan by 256-128-64 nöron mimarisi
# ve 0.3-0.3-0.3 dropout değerleri ile gerçekleştirilecek
!python src/main.py

# Not: Bayesian optimizasyon kullanarak da eğitim yapabilirsiniz.
# Varsayılan olarak en iyi performans gösteren standart model mimarisi kullanılır.
