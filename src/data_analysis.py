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
    
    # Her bant için istatistikler
    for band in bands:
        min_val = data[band].min()
        max_val = data[band].max()
        mean_val = data[band].mean()
        median_val = data[band].median()
        print(f"{band}-band: Min={min_val:.2f}, Max={max_val:.2f}, Ortalama={mean_val:.2f}, Medyan={median_val:.2f}")
    
    for i, band in enumerate(bands, 1):
        plt.subplot(2, 3, i)
        
        # Minimum ve maksimum değerleri hesapla
        min_val = data[band].min()
        max_val = data[band].max()
        
        # Aykırı değerler için filtreleme (i-band için önemli)
        q1 = data[band].quantile(0.01)  # Alt %1
        q3 = data[band].quantile(0.99)  # Üst %1
        filtered_data = data[band][(data[band] >= q1) & (data[band] <= q3)]
        
        # Histogram çiz (mümkünse filtrelenmiş veriyle)
        sns.histplot(filtered_data, kde=True, bins=30, color='blue')
        
        # Eksen limitlerini ayarla
        if band == 'i':
            # Filtrelenmiş veriye göre limitler belirle
            plt.xlim(filtered_data.min() - 0.1, filtered_data.max() + 0.1)
        
        plt.title(f"{band}-Band Parlaklık Dağılımı\nMin: {min_val:.2f}, Max: {max_val:.2f}\nFiltreli: {filtered_data.min():.2f} - {filtered_data.max():.2f}")
        plt.xlabel("Parlaklık (mag)")
        plt.ylabel("Frekans")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "magnitude_distributions.png"))
    plt.close()

def analyze_band_data(data, band_name):
    """Belirli bir parlaklık bandını detaylı olarak analiz eder."""
    band_data = data[band_name]
    
    print(f"\n=== {band_name}-band Detaylı Analiz ===")
    print(f"Veri türü: {band_data.dtype}")
    print(f"Toplam veri sayısı: {len(band_data)}")
    print(f"Boş değer sayısı: {band_data.isna().sum()}")
    print(f"Benzersiz değer sayısı: {band_data.nunique()}")
    
    # Minimum değerleri bul
    min_values = band_data.nsmallest(5)
    print(f"\nEn küçük 5 değer:")
    for idx, val in min_values.items():
        print(f"  Indeks {idx}: {val}")
    
    # Maksimum değerleri bul
    max_values = band_data.nlargest(5)
    print(f"\nEn büyük 5 değer:")
    for idx, val in max_values.items():
        print(f"  Indeks {idx}: {val}")
    
    # Sıfıra yakın değerleri bul (i-band sorunu için)
    near_zero = band_data[band_data.between(-0.1, 0.1)]
    print(f"\nSıfıra yakın değerler: {len(near_zero)} adet")
    
    # Aykırı değerleri filtrele
    q1 = band_data.quantile(0.25)
    q3 = band_data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = band_data[(band_data < lower_bound) | (band_data > upper_bound)]
    print(f"\nAykırı değerler: {len(outliers)} adet")
    print(f"Aykırı değer yüzdesi: {len(outliers)/len(band_data)*100:.2f}%")
    
    # Histogram çiz (aykırı değerler olmadan)
    plt.figure(figsize=(10, 6))
    normal_data = band_data[(band_data >= lower_bound) & (band_data <= upper_bound)]
    sns.histplot(normal_data, kde=True, bins=30, color='green')
    plt.title(f"{band_name}-Band Parlaklık Dağılımı (Aykırı Değerler Hariç)")
    plt.xlabel("Parlaklık (mag)")
    plt.ylabel("Frekans")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{band_name}_band_filtered.png"))
    plt.close()
    print(f"Aykırı değerler olmadan histogram kaydedildi: {band_name}_band_filtered.png")

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
        
        # i-band için detaylı analiz
        analyze_band_data(data, 'i')
    else:
        print("Veri yüklenemedi. Analiz yapılamadı.")

if __name__ == "__main__":
    main()
