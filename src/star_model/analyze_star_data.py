import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from prepare_data import load_star_subset

def main():
    # Veriyi yükle
    data_path_star = 'data/star_subtypes.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, le_star, scaler_star = load_star_subset(data_path_star)
    
    # Ham veriyi de yükle (analiz için)
    df = pd.read_csv(data_path_star, encoding="utf-8")
    star_df = df[df["class"] == "STAR"].copy()
    
    # 1) Sınıf dağılımını görselleştir
    plt.figure(figsize=(12, 6))
    subclass_counts = star_df["subClass"].value_counts()
    ax = sns.barplot(x=subclass_counts.index, y=subclass_counts.values)
    plt.title('Yıldız Alt Tür Dağılımı')
    plt.xlabel('Alt Tür')
    plt.ylabel('Örnek Sayısı')
    plt.xticks(rotation=45)
    
    # Çubukların üzerine değerleri yaz
    for i, v in enumerate(subclass_counts.values):
        ax.text(i, v + 10, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('outputs/star_subclass_distribution.png')
    plt.close()
    
    # 2) Özellikler arası korelasyonu görselleştir
    # Renk indekslerini hesapla
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        star_df[f"{a}_{b}"] = star_df[a] - star_df[b]
    
    # Sadece renk/magnitude özelliklerini seçelim
    features = ['u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z']
    corr = star_df[features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Özellikler Arası Korelasyon')
    plt.tight_layout()
    plt.savefig('outputs/feature_correlation.png')
    plt.close()
    
    # 3) PCA ile boyut indirgeme ve görselleştirme
    # Eğitim, doğrulama ve test verilerini birleştir
    X_combined = np.vstack([X_train, X_val, X_test])
    y_indices = np.argmax(np.vstack([y_train, y_val, y_test]), axis=1)
    y_labels = le_star.inverse_transform(y_indices)
    
    # PCA uygula
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    # PCA sonuçlarını görselleştir
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(np.unique(y_labels)):
        plt.scatter(
            X_pca[y_labels == class_name, 0],
            X_pca[y_labels == class_name, 1],
            label=class_name,
            alpha=0.7
        )
    
    plt.title('PCA: Yıldız Alt Türleri')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varyans)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varyans)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/pca_visualization.png')
    plt.close()
    
    # 4) t-SNE ile boyut indirgeme ve görselleştirme
    # Daha az örnek seçelim (t-SNE yavaş çalışabilir)
    sample_size = min(3000, len(X_combined))
    indices = np.random.choice(len(X_combined), sample_size, replace=False)
    X_sample = X_combined[indices]
    y_sample = y_labels[indices]
    
    # t-SNE uygula
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_sample)
    
    # t-SNE sonuçlarını görselleştir
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(np.unique(y_sample)):
        plt.scatter(
            X_tsne[y_sample == class_name, 0],
            X_tsne[y_sample == class_name, 1],
            label=class_name,
            alpha=0.7
        )
    
    plt.title('t-SNE: Yıldız Alt Türleri')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/tsne_visualization.png')
    
    # 5) Önemli özellikleri belirle
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, np.argmax(y_train, axis=1))
    
    # Özellik önemliliklerini hesapla
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Özellik isimleri (prepare_data.py'de oluşturulan özellikler)
    feature_names = list(df.columns)
    # Kullanılmayan sütunları çıkar
    for col in ["class", "subClass", "objid", "specobjid", "run", "rerun", "camcol", "field"]:
        if col in feature_names:
            feature_names.remove(col)
    
    # Renk indeksi ve diğer oluşturulan özellikleri ekle
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        feature_names.append(f"{a}_{b}")
      # Top 10 özelliği görselleştir
    plt.figure(figsize=(10, 6))
    plt.title("Özellik Önemlilikleri")
    
    # İndeksleri feature_names boyutuna göre filtreleyelim
    valid_indices = [i for i in indices if i < len(feature_names)]
    top_n = min(10, len(valid_indices))
    
    plt.bar(range(top_n), importances[valid_indices[:top_n]], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in valid_indices[:top_n]], rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    
    print("Analiz tamamlandı! Grafikler 'outputs' klasörüne kaydedildi.")

if __name__ == "__main__":
    main()