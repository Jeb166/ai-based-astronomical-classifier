import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Veri dosyasının yolu
DATA_PATH = "c:/Users/Emre/Desktop/ai-based-astronomical-classifier/data/skyserver.csv"
OUTPUT_DIR = "c:/Users/Emre/Desktop/ai-based-astronomical-classifier/outputs"

# Çıktı klasörünü oluştur
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Veriyi yükler."""
    try:
        data = pd.read_csv(DATA_PATH)
        return data
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return None

def plot_class_distribution(data):
    """Sınıf dağılımını çizer."""
    plt.figure(figsize=(8, 6))
    # FutureWarning'i önlemek için güncellenmiş kullanım
    sns.countplot(data=data, x='class', hue='class', palette='viridis', legend=False)
    plt.title("Sınıf Dağılımı")
    plt.xlabel("Sınıf")
    plt.ylabel("Sayı")
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
    plt.close()

def plot_feature_correlation(data):
    """Özellikler arasındaki korelasyonu çizer."""
    # Sadece numerik sütunları seç
    numeric_data = data.select_dtypes(include=['number'])
    plt.figure(figsize=(12, 10))
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Özellik Korelasyon Matrisi")
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"))
    plt.close()

def plot_magnitude_distributions(data):
    """Parlaklık (magnitude) dağılımlarını çizer."""
    bands = ['u', 'g', 'r', 'i', 'z']
    plt.figure(figsize=(15, 10))
    for i, band in enumerate(bands, 1):
        plt.subplot(2, 3, i)
        sns.histplot(data[band], kde=True, bins=30, color='blue')
        plt.title(f"{band}-Band Parlaklık Dağılımı")
        plt.xlabel("Parlaklık (mag)")
        plt.ylabel("Frekans")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "magnitude_distributions.png"))
    plt.close()

def main():
    """Ana analiz fonksiyonu."""
    data = load_data()
    if data is not None:
        print("Veri başarıyla yüklendi. Analiz başlıyor...")
        plot_class_distribution(data)
        print("Sınıf dağılımı grafiği oluşturuldu.")
        plot_feature_correlation(data)
        print("Özellik korelasyon grafiği oluşturuldu.")
        plot_magnitude_distributions(data)
        print("Parlaklık dağılım grafikleri oluşturuldu.")
    else:
        print("Veri yüklenemedi. Analiz yapılamadı.")

if __name__ == "__main__":
    main()
