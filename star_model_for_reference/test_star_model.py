import os
import time
import numpy as np
import tensorflow as tf
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
    
    # Test edilecek tüm model türleri
    model_types = [
        ("Hafif", "lightweight"),
        ("Standart", "standard"),
        ("Ayrılabilir", "separable"),
        ("Ağaç Yapılı", "tree")
    ]
    
    # Her modeli test et ve sonuçları sakla
    results = []
    
    for model_name, model_type in model_types:
        print(f"\n{model_name.upper()} MODEL TESTİ")
        print("-" * (len(model_name) + 12))
        start_time = time.time()
        
        # Modeli oluştur
        model = build_star_model(
            n_features, n_classes, 
            model_type=model_type,
            neurons1=256,
            neurons2=128,
            dropout1=0.3,
            dropout2=0.3,
            learning_rate=0.001
        )
        
        # Gelişmiş stratejileri kullanarak eğit
        model, history = train_star_model(
            model, Xs_tr, ys_tr, Xs_val, ys_val, 
            class_weights=cw_dict, 
            max_samples=30000,
            batch_size=128,
            epochs=20,
            use_cyclic_lr=True,
            use_trending_early_stop=True
        )
        
        # Eğitim süresini ve test doğruluğunu hesapla
        training_time = time.time() - start_time
        test_acc = (model.predict(Xs_te).argmax(1) == ys_te.argmax(1)).mean() * 100
        
        # Sonuçları yazdır
        print(f"\n{model_name} model eğitim süresi: {training_time:.2f} saniye")
        print(f"{model_name} model test doğruluğu: {test_acc:.2f}%")
        
        # Sonuçları kaydet
        results.append({
            'name': model_name,
            'type': model_type,
            'accuracy': test_acc,
            'time': training_time,
            'model': model
        })
    
    # Model karşılaştırma özeti
    print("\nMODEL KARŞILAŞTIRMASI")
    print("-------------------")
    
    for result in results:
        print(f"{result['name']} Model: "
              f"{result['accuracy']:.2f}% doğruluk, "
              f"{result['time']:.2f} saniye")
    
    # Doğruluk bazlı en iyi model
    best_accuracy_model = max(results, key=lambda x: x['accuracy'])
    print(f"\nEn yüksek doğruluk: {best_accuracy_model['name']} "
          f"({best_accuracy_model['accuracy']:.2f}%)")
    
    # Hız bazlı en iyi model
    sorted_by_time = sorted(results, key=lambda x: x['time'])
    fastest_model = sorted_by_time[0]
    print(f"En hızlı eğitim: {fastest_model['name']} "
          f"({fastest_model['time']:.2f} saniye)")
    
    # Doğruluk/hız dengesi
    for i, result in enumerate(results):
        result['speed_score'] = (result['accuracy'] / 
                               (result['time'] / min(r['time'] for r in results)))
    
    best_balanced_model = max(results, key=lambda x: x['speed_score'])
    print(f"En iyi denge: {best_balanced_model['name']} "
          f"(Hız skoru: {best_balanced_model['speed_score']:.2f})")
    
    # En iyi modeli kaydet (doğruluk bazlı veya dengeye göre seçilebilir)
    best_model = best_accuracy_model['model']  # veya best_balanced_model
    best_model.save(f"{out_dir}/optimized_star_model.keras")
    print(f"\nEn iyi model kaydedildi: {out_dir}/optimized_star_model.keras")
    
    # Tercihen tüm modelleri de kaydedebiliriz
    for result in results:
        model_path = f"{out_dir}/{result['type']}_star_model.keras"
        result['model'].save(model_path)
        print(f"{result['name']} model kaydedildi: {model_path}")

if __name__ == "__main__":
    # GPU kullanımını optimize et
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # GPU bellek büyümesini yalnızca gerektikçe ayarla
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()
