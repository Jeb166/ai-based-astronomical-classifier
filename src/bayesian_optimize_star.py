import os
import numpy as np
import time
from functools import partial
from sklearn.utils import class_weight
import tensorflow as tf

# Skopt için kütüphaneler
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

# Veri ve model için kütüphaneler
from prepare_data import load_star_subset
from star_model import build_star_model, train_star_model
from download_data import check_data_files

# Optimizasyon fonksiyonumuz
def optimize_star_model_bayesian(n_trials=15, save_dir='outputs'):
    """
    Bayesian optimizasyonu kullanarak star modeli hiperparametrelerini optimizasyon yapar
    
    Parametreler:
    - n_trials: Deneme sayısı
    - save_dir: Çıktıların kaydedileceği dizin
    
    Returns:
    - En iyi model
    - En iyi parametreler
    - Optimizasyon sonuçları
    """
    # Çıktı dizinini oluştur
    os.makedirs(save_dir, exist_ok=True)
    
    # Veri dosyalarını kontrol et
    if not check_data_files():
        print("Veri dosyaları eksik. Optimizasyon iptal ediliyor.")
        print("Lütfen eksik veri dosyalarını 'data/' klasörüne yükleyin ve tekrar deneyin.")
        return None, None, None
            try:
                # download_data modülünü dinamik olarak yükle
                spec = importlib.util.spec_from_file_location("download_data", download_module_path)
                download_data = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(download_data)
                
                # Veri dosyalarını indir
                download_data.check_and_download_data()
                
                # Tekrar kontrol et
                all_found = True
                for file in missing_files:
                    if not os.path.exists(os.path.join(data_dir, file)):
                        all_found = False
                        print(f"Uyarı: {file} hala eksik.")
                
                if not all_found:
                    print("\nBazı veri dosyaları hala eksik. Lütfen bunları manuel olarak yükleyin.")
                    print("Colab'da dosyaları manuel olarak yüklemek için şu adımları izleyin:")
                    print("1. Sol paneldeki dosya simgesine tıklayın")
                    print("2. data/ klasörüne tıklayın (yoksa oluşturun)")
                    print("3. Yükle butonunu kullanarak dosyaları yükleyin")
                    print("\nDosyalar yüklendikten sonra programı tekrar çalıştırın.")
                    return False
            except Exception as e:
                print(f"Veri indirme hatası: {e}")
                print("\nLütfen veri dosyalarını manuel olarak data/ dizinine yükleyin.")
                return False
        else:
            print("\nLütfen veri dosyalarını manuel olarak data/ dizinine yükleyin.")
            return False
    
    return True

# Optimizasyon fonksiyonumuz
def optimize_star_model_bayesian(n_trials=15, save_dir='outputs'):
    """
    Bayesian optimizasyonu kullanarak star modeli hiperparametrelerini optimizasyon yapar
    
    Parametreler:
    - n_trials: Deneme sayısı
    - save_dir: Çıktıların kaydedileceği dizin
    
    Returns:
    - En iyi model
    - En iyi parametreler
    - Optimizasyon sonuçları
    """
    # Çıktı dizinini oluştur
    os.makedirs(save_dir, exist_ok=True)
    
    # Veri dosyalarını kontrol et
    if not check_data_files():
        print("Veri dosyaları eksik. Optimizasyon iptal ediliyor.")
        return None, None, None
      # Veriyi yükle
    print("Veri yükleniyor...")
    data_path_star = 'data/star_subtypes.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, le_star, scaler_star = load_star_subset(data_path_star)
    
    # Sınıf ağırlıklarını hesapla
    y_int = y_train.argmax(1)
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_int), y=y_int)
    cw_dict = dict(enumerate(cw))
    
    # Model boyutları
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]
    
    print(f"Özellik sayısı: {n_features}, Sınıf sayısı: {n_classes}")
    
    # Parametre uzayını tanımla
    param_space = [
        Categorical(['standard', 'lightweight', 'separable', 'tree'], name='model_type'),
        Integer(8, 32, name='rank'),  # Separable model için rank
        Integer(128, 512, name='neurons1'),
        Integer(64, 256, name='neurons2'),
        Real(0.2, 0.5, name='dropout1'),
        Real(0.2, 0.5, name='dropout2'),
        Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
        Integer(64, 256, name='batch_size')
    ]
      # Optimizasyon için objektif fonksiyon
    @use_named_args(param_space)
    def objective(model_type, rank, neurons1, neurons2, dropout1, dropout2, learning_rate, batch_size):
        """
        Her hiperparametre seti için modeli eğitir ve doğrulama doğruluğunu döndürür.
        Minimum'a optimizasyon yaptığımız için negatif doğruluk döndürürüz.
        """
        params = {
            'model_type': model_type,
            'rank': rank,
            'neurons1': neurons1,
            'neurons2': neurons2,
            'dropout1': dropout1,
            'dropout2': dropout2,
            'learning_rate': learning_rate,
            'batch_size': batch_size
        }        print(f"\nParametreleri değerlendiriyorum: {params}")
        
        # Model için parametreleri hazırla (batch_size hariç)
        model_params = {k: v for k, v in params.items() if k != 'batch_size'}
        
        # Modeli oluştur - batch_size parametresini ÇIKARIYORUZ
        model = build_star_model(
            input_dim=n_features, 
            n_classes=n_classes,
            **model_params  # batch_size bu listeye dahil edilmiyor
        )
        
        # Eğitim için daha az veri kullan (hızlı değerlendirme için)
        max_samples = 20000
        
        start_time = time.time()
        
        # Modeli eğit
        model, history = train_star_model(
            model, 
            X_train, y_train, 
            X_val, y_val, 
            class_weights=cw_dict,
            max_samples=max_samples,
            batch_size=batch_size,
            epochs=15,  # Daha az epoch
            use_cyclic_lr=True,
            use_trending_early_stop=True
        )
        
        training_time = time.time() - start_time
        
        # Doğrulama doğruluğunu hesapla
        val_accuracy = max(history.history['val_categorical_accuracy'])
        
        print(f"Model tipi: {params['model_type']}, "
              f"Doğrulama doğruluğu: {val_accuracy:.4f}, "
              f"Eğitim süresi: {training_time:.1f} saniye")
          # Minimizasyon için negatif doğruluk
        return -val_accuracy
    
    # Bayesian optimizasyonu başlat
    print(f"\nBayesian optimizasyon başlıyor... {n_trials} deneme yapılacak")
    
    result = gp_minimize(
        objective,
        param_space,
        n_calls=n_trials,
        n_random_starts=5,  # İlk 5 rastgele, sonrakiler Bayesian
        random_state=42,
        verbose=True
    )
    
    # En iyi parametreleri yazdır
    best_params = dict(zip([dim.name for dim in param_space], result.x))
    print("\nEn iyi parametreler:")
    for param, value in best_params.items():
        print(f"{param}: {value}")    # En iyi modeli eğit (tam verilerle)
    print("\nEn iyi model tam verilerle eğitiliyor...")
    
    # batch_size parametresini build_star_model fonksiyonu için çıkarıyoruz
    model_params = {k: v for k, v in best_params.items() if k != 'batch_size'}
    
    best_model = build_star_model(
        input_dim=n_features, 
        n_classes=n_classes,
        **model_params  # batch_size bu parametrelerin içinde olmayacak
    )
    
    best_model, _ = train_star_model(
        best_model, 
        X_train, y_train, 
        X_val, y_val, 
        class_weights=cw_dict,
        batch_size=best_params['batch_size'],
        epochs=20,
        use_cyclic_lr=True,
        use_trending_early_stop=True
    )
    
    # Test doğruluğunu hesapla
    test_preds = best_model.predict(X_test)
    test_accuracy = (test_preds.argmax(1) == y_test.argmax(1)).mean() * 100
    print(f"\nTest doğruluğu: {test_accuracy:.2f}%")
    
    # En iyi modeli kaydet
    model_path = os.path.join(save_dir, "bayesian_optimized_star_model.keras")
    best_model.save(model_path)
    print(f"En iyi model kaydedildi: {model_path}")
    
    # Optimizasyon sonuçlarını da kaydet
    import joblib
    result_path = os.path.join(save_dir, "bayesian_optimization_results.joblib")
    joblib.dump(result, result_path)
    print(f"Optimizasyon sonuçları kaydedildi: {result_path}")
    
    # Grafikleri oluştur ve kaydet
    try:
        import matplotlib.pyplot as plt
        
        # Yakınsama grafiği
        plt.figure(figsize=(10, 6))
        plot_convergence(result)
        plt.title("Bayesian Optimizasyon Yakınsama Grafiği")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bayesian_convergence.png"), dpi=150)
        
        # En önemli parametreleri göster
        fig, ax = plt.subplots(1, figsize=(12, 8))
        plot_objective(result, dimensions=['model_type', 'neurons1', 'dropout1', 'learning_rate'])
        plt.title("Bayesian Optimizasyon - Parametre Önemlilikleri")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "bayesian_parameter_importance.png"), dpi=150)
        
        print("Grafikler outputs klasörüne kaydedildi.")
    except Exception as e:
        print(f"Grafik oluşturmada hata: {e}")
    
    return best_model, best_params, result

if __name__ == "__main__":
    # GPU kullanımını optimize et
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # GPU bellek büyümesini ayarla
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU kullanıma hazır: {len(gpus)} adet GPU bulundu")
        except RuntimeError as e:
            print(f"GPU ayarlanırken hata: {e}")
    
    # Veri dosyalarını kontrol et
    if check_data_files():
        # Bayesian optimizasyonu çalıştır
        optimize_star_model_bayesian(n_trials=15)
    else:
        print("\nOptimizasyon iptal edildi. Lütfen eksik veri dosyalarını 'data/' klasörüne yükleyin.")
        print("Dosyaları yükledikten sonra programı tekrar çalıştırın.")
