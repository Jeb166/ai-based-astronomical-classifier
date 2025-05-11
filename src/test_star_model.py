import os
import time
import numpy as np
from sklearn.utils import class_weight
from prepare_data import load_star_subset
from star_model import build_star_model, train_star_model

def main():
    # Çıktı dizini oluştur
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    
    # Veriyi yükle
    print("Yıldız alt tür verileri yükleniyor...")
    data_path_star = 'data/star_subtypes.csv'
    Xs_tr, Xs_val, Xs_te, ys_tr, ys_val, ys_te, le_star, scaler_star = load_star_subset(data_path_star)
    
    # Sınıf ağırlıklarını hesapla
    y_int = ys_tr.argmax(1)
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_int), y=y_int)
    cw_dict = dict(enumerate(cw))
    
    # Model boyutları
    n_features = Xs_tr.shape[1]
    n_classes = ys_tr.shape[1]
    
    print(f"Özellik sayısı: {n_features}, Sınıf sayısı: {n_classes}")
    print(f"Eğitim örneği sayısı: {len(Xs_tr)}")
    
    # Hafif model oluştur
    print("\n1. HAFİF MODEL TESTİ")
    print("--------------------")
    start_time = time.time()
    
    light_model = build_star_model(n_features, n_classes, lightweight=True)
    light_model, history = train_star_model(
        light_model, Xs_tr, ys_tr, Xs_val, ys_val, 
        class_weights=cw_dict, max_samples=30000
    )
    
    light_time = time.time() - start_time
    light_test_acc = (light_model.predict(Xs_te).argmax(1)==ys_te.argmax(1)).mean()*100
    
    print(f"\nHafif model eğitim süresi: {light_time:.2f} saniye")
    print(f"Hafif model test doğruluğu: {light_test_acc:.2f}%")
    
    # Karşılaştırma için standart model
    print("\n2. STANDART MODEL TESTİ")
    print("----------------------")
    start_time = time.time()
    
    standard_model = build_star_model(n_features, n_classes, lightweight=False)
    standard_model, history = train_star_model(
        standard_model, Xs_tr, ys_tr, Xs_val, ys_val, 
        class_weights=cw_dict, max_samples=30000
    )
    
    standard_time = time.time() - start_time
    standard_test_acc = (standard_model.predict(Xs_te).argmax(1)==ys_te.argmax(1)).mean()*100
    
    print(f"\nStandart model eğitim süresi: {standard_time:.2f} saniye")
    print(f"Standart model test doğruluğu: {standard_test_acc:.2f}%")
    
    # Karşılaştırma özeti
    print("\nMODEL KARŞILAŞTIRMASI")
    print("-------------------")
    print(f"Hafif model:    {light_test_acc:.2f}% doğruluk, {light_time:.2f} saniye")
    print(f"Standart model: {standard_test_acc:.2f}% doğruluk, {standard_time:.2f} saniye")
    print(f"Hız artışı:     {standard_time/light_time:.2f}x daha hızlı")
    print(f"Doğruluk farkı: {standard_test_acc-light_test_acc:.2f}%")
    
    # En iyi modeli kaydet
    best_model = light_model if light_test_acc >= standard_test_acc else standard_model
    best_model.save(f"{out_dir}/optimized_star_model.keras")
    print(f"\nEn iyi model kaydedildi: {out_dir}/optimized_star_model.keras")

if __name__ == "__main__":
    main()
