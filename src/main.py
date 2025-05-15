import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

from prepare_data import load_and_prepare, load_star_subset
from model import build_model
from star_model import build_star_model, train_star_model

# ------------------------------------------------------------------
# Helper will be defined after models are trained
# ------------------------------------------------------------------

def main():
    # GPU kullanımını optimize et
    print("TensorFlow sürümü:", tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # GPU bellek büyümesini ayarla
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU kullanıma hazır: {len(gpus)} adet GPU bulundu")
        except RuntimeError as e:
            print(f"GPU ayarlanırken hata: {e}")
    else:
        print("GPU bulunamadı, CPU kullanılacak.")
    
    # ------------------------------------------------------------------
    # 0) Paths
    # ------------------------------------------------------------------
    data_path       = 'data/skyserver.csv'
    data_path_star  = 'data/star_subtypes.csv'
    out_dir         = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) LOAD GAL/QSO/STAR DATA
    # ------------------------------------------------------------------
    X_tr, X_val, X_te, y_tr, y_val, y_te, df_full = load_and_prepare(data_path)
    y_tr_lbl  = y_tr.argmax(1)
    y_val_lbl = y_val.argmax(1)
    y_te_lbl  = y_te.argmax(1)

    # ------------------------------------------------------------------
    # 2) DNN
    # ------------------------------------------------------------------
    dnn = build_model(X_tr.shape[1], y_tr.shape[1])
    dnn.fit(
        X_tr, y_tr,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                   ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)],
        verbose=1
    )

    # ------------------------------------------------------------------
    # 3) RANDOM FOREST
    # ------------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        oob_score=True)
    rf.fit(X_tr, y_tr_lbl)
    print(f"RF OOB accuracy: {rf.oob_score_:.4f}")

    # ------------------------------------------------------------------
    # 4) VALIDATION‑BASED BEST WEIGHT
    # ------------------------------------------------------------------
    dnn_val = dnn.predict(X_val)
    rf_val  = rf.predict_proba(X_val)
    best_w, best_acc = 0.5, 0.0
    for w in np.linspace(0.1, 0.9, 9):
        if (proba_acc := ((w*dnn_val+(1-w)*rf_val).argmax(1)==y_val_lbl).mean()) > best_acc:
            best_w, best_acc = w, proba_acc
    print(f"[Val] Best DNN weight: {best_w:.2f}  (val acc={best_acc*100:.2f}%)")

    # ------------------------------------------------------------------
    # 5) TEST SET SCORES
    # ------------------------------------------------------------------
    dnn_probs = dnn.predict(X_te)
    rf_probs  = rf.predict_proba(X_te)
    ens_probs = best_w*dnn_probs + (1-best_w)*rf_probs
    dnn_acc = (dnn_probs.argmax(1)==y_te_lbl).mean()*100
    rf_acc  = (rf_probs.argmax(1)==y_te_lbl).mean()*100
    ens_acc = (ens_probs.argmax(1)==y_te_lbl).mean()*100
    print(f"DNN  Test Accuracy : {dnn_acc:6.3f}%")
    print(f"RF   Test Accuracy : {rf_acc :6.3f}%")
    print(f"BEST‑W ENS Accuracy : {ens_acc:6.3f}%")

    labels = np.unique(df_full['class'])
    sns.heatmap(confusion_matrix(y_te_lbl, ens_probs.argmax(1)), annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix — Best‑W Ensemble')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_ens_bestw.png", dpi=150)
    plt.show()

    # ------------------------------------------------------------------
    # 6) SAVE GAL/QSO/STAR MODELS
    # ------------------------------------------------------------------
    dnn.save(f"{out_dir}/dnn_model.keras")
    import joblib
    joblib.dump(rf, f"{out_dir}/rf_model.joblib")
    
    # ------------------------------------------------------------------
    # 7) STAR SUB‑CLASS MODEL
    # ------------------------------------------------------------------    # A) Yıldız veri setini yükle
    print("\n" + "="*70)
    print("YILDIZ ALT TÜR MODELİ EĞİTİMİ VE OPTİMİZASYONU".center(70))
    print("="*70)
    
    # Veri yükleme
    try:
        Xs_tr, Xs_val, Xs_te, ys_tr, ys_val, ys_te, le_star, scaler_star = load_star_subset(data_path_star)
        
        # Veri kontrolü - NaN/Inf değerleri tespit için
        print("Veri kontrol ediliyor...")
        nan_count_train = np.isnan(Xs_tr).sum()
        inf_count_train = np.isinf(Xs_tr).sum()
        if nan_count_train > 0 or inf_count_train > 0:
            print(f"UYARI: Eğitim verisinde {nan_count_train} NaN ve {inf_count_train} Inf değer bulundu.")
            print("Bu değerler otomatik olarak temizlenecek.")
            # Ekstra güvenlik - NaN'ları temizle
            Xs_tr = np.nan_to_num(Xs_tr, nan=0.0, posinf=0.0, neginf=0.0)
            Xs_val = np.nan_to_num(Xs_val, nan=0.0, posinf=0.0, neginf=0.0)
            Xs_te = np.nan_to_num(Xs_te, nan=0.0, posinf=0.0, neginf=0.0)
          # Etiketlerin durumunu kontrol et ve güvenli bir şekilde dönüştür
        print("\nYıldız alt tür etiketlerini hazırlıyorum...")
        # ys_tr doğrudan sayısal one-hot encoded format mı?
        if isinstance(ys_tr, np.ndarray) and len(ys_tr.shape) == 2:
            print("Etiketler zaten one-hot kodlanmış.")
            y_int = np.argmax(ys_tr, axis=1)
        # Yoksa string veya sayısal sınıf değerlerini içeren bir dizi mi?
        else:
            # Kategorik string değerleri ise LabelEncoder ile dönüştür
            if not np.issubdtype(ys_tr.dtype, np.number):
                print("Kategorik etiketleri sayısal değerlere dönüştürüyorum...")
                y_int = le_star.transform(ys_tr)
            else:
                print("Etiketler zaten sayısal format.")
                y_int = ys_tr
        
        # Sınıf ağırlıklarını hesapla
        unique_classes = np.unique(y_int)
        print(f"Benzersiz sınıf sayısı: {len(unique_classes)}")
        print(f"Örnek etiketler: {y_int[:5]}")
        
        # Class weights hesapla
        cw = class_weight.compute_class_weight("balanced", classes=unique_classes, y=y_int)
        cw_dict = dict(zip(unique_classes, cw))
        
        # B) Yıldız Modeli Eğitimi
        print("\nYILDIZ ALT TÜR MODELİ EĞİTİMİ")
        print("-" * 40)
        
        # Model boyutları
        n_features = Xs_tr.shape[1]
        n_classes = ys_tr.shape[1]
        
        print("Yıldız modeli eğitiliyor...")
        print(f"Özellik sayısı: {n_features}, Sınıf sayısı: {n_classes}")
        
        # Modeli oluştur - optimize edilmiş parametreleri kullan
        star_net = build_star_model(
            n_features, n_classes,
            neurons1=490,         # Arttırıldı: 434 -> 490
            neurons2=120,         # Arttırıldı: 99 -> 120
            neurons3=120,         # Arttırıldı: 107 -> 120
            dropout1=0.4,         # Ayarlandı: 0.379 -> 0.4
            dropout2=0.35,        # Ayarlandı: 0.334 -> 0.35
            dropout3=0.25,        # Ayarlandı: 0.230 -> 0.25
            learning_rate=0.0002  # Ayarlandı: 0.00024 -> 0.0002
        )
        
        # Gelişmiş stratejileri kullanarak eğit
        star_net, history = train_star_model(
            star_net, Xs_tr, ys_tr, Xs_val, ys_val, 
            class_weights=cw_dict,
            batch_size=64,        # Daha küçük: 128 -> 64 (daha iyi genelleme)
            epochs=30,            # Daha uzun: 20 -> 30
            use_cyclic_lr=True,
            use_trending_early_stop=True
        )
        
        # Test doğruluğunu hesapla
        y_pred = star_net.predict(Xs_te)
        star_acc = (y_pred.argmax(1) == ys_te.argmax(1)).mean() * 100
        print(f"\nYıldız alt türleri test doğruluğu: {star_acc:.2f}%")
        
        # Confusion matrix - yıldız alt türleri
        plt.figure(figsize=(10, 8))
        y_pred_classes = y_pred.argmax(1)
        y_true_classes = ys_te.argmax(1)
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le_star.classes_, yticklabels=le_star.classes_)
        plt.title('Yıldız Alt Türleri - Confusion Matrix')
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.tight_layout()
        plt.savefig(f"{out_dir}/star_subtypes_confusion.png", dpi=150)
        plt.show()
        
        # Modeli kaydet
        star_net.save(f"{out_dir}/star_model.keras")
        joblib.dump(le_star, f"{out_dir}/star_label_enc.joblib")
        joblib.dump(scaler_star, f"{out_dir}/star_scaler.joblib")
        
    except Exception as e:
        print(f"Yıldız modeli eğitiminde hata: {str(e)}")
        print("Yıldız alt tür modeli eğitimi atlanıyor.")
        # Boş değerler ata ki ilerideki kod çalışsın
        star_net, le_star, scaler_star = None, None, None
    
    # ------------------ helper for UI / further use -----------------
    global full_predict
    def full_predict(sample_array):
        """Return GALAXY/QSO/STAR‑subtype for each input row (2‑stage)."""
        p_ens = best_w*dnn.predict(sample_array) + (1-best_w)*rf.predict_proba(sample_array)
        primary = p_ens.argmax(1)
        STAR_ID = np.where(labels=='STAR')[0][0]
        out = []
        for i, cls in enumerate(primary):
            if cls == STAR_ID:
                x_s = scaler_star.transform(sample_array[i:i+1])
                sub_id = star_net.predict(x_s).argmax(1)[0]
                out.append(f"STAR-{le_star.inverse_transform([sub_id])[0]}")
            else:
                out.append(labels[cls])
        return out

    # leave models in global scope for interactive sessions
    globals().update({'dnn': dnn, 'rf': rf, 'best_w': best_w,
                      'labels': labels, 'star_net': star_net,
                      'scaler_star': scaler_star, 'le_star': le_star})
    
    print("\n" + "="*70)
    print("TÜM EĞİTİM VE OPTİMİZASYON İŞLEMLERİ TAMAMLANDI".center(70))
    print("="*70)
    print(f"\nTüm modeller '{out_dir}' klasörüne kaydedildi.")
    print("\nÖnemli dosyalar:")
    print(f"- {out_dir}/dnn_model.keras: Ana DNN modeli")
    print(f"- {out_dir}/rf_model.joblib: Random Forest modeli")
    print(f"- {out_dir}/star_model.keras: Yıldız alt tür modeli")
    
    # Modelinizi kullanmak için:    print("\nBu modeli kullanmak için:")
    print(">>> from main import full_predict")
    print(">>> sonuclar = full_predict(yeni_veriler)")

def run_advanced_star_model():
    """Gelişmiş yıldız modeli eğitimi ve optimizasyon seçeneklerini çalıştır"""
    try:
        # scikit-optimize kütüphanesini kontrol et ve yükle
        try:
            import skopt
        except ImportError:
            print("\nBayesian optimizasyon için 'scikit-optimize' kütüphanesi yükleniyor...")
            import sys
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize"])
            print("scikit-optimize başarıyla yüklendi!")
            
        # Model eğitimi için iki seçenek sun
        print("\n" + "="*70)
        print("GELİŞMİŞ YILDIZ MODELİ EĞİTİM SEÇENEKLERİ".center(70))
        print("="*70)
        print("1. Dengeli parametrelerle modeli eğit (daha kararlı sonuçlar)")
        print("2. Bayesian optimizasyon ile yeni parametreler bul (deneysel)")
        try:
            option = int(input("\nSeçiminiz (1/2) [varsayılan=1]: ") or "1")
        except ValueError:
            option = 1
            print("Geçersiz seçim, varsayılan olarak mevcut parametrelerle eğitilecek.")
        
        if option == 2:
            # Tam optimizasyon çalıştırmak istiyorsa
            from bayesian_optimize_star import optimize_star_model_bayesian
            print("\n" + "="*70)
            print("BAYESIAN OPTİMİZASYON BAŞLATILIYOR".center(70))
            print("="*70)
            
            print("\nYıldız modeli için Bayesian optimizasyon çalıştırılıyor...")
            model, best_params, result = optimize_star_model_bayesian(n_trials=10, save_dir='outputs')
            
            # En iyi parametreleri raporla
            if best_params:
                print("\nEN İYİ HİPERPARAMETRELER:")
                print("-" * 40)
                for param, value in best_params.items():
                    print(f"{param}: {value}")
                
                print(f"\nOptimize edilmiş model kaydedildi: outputs/bayesian_optimized_star_model.keras")
                print("Bu modeli kullanmak için:")
                print(">>> from tensorflow.keras.models import load_model")
                print(">>> model = load_model('outputs/bayesian_optimized_star_model.keras')")
                print(">>> tahminler = model.predict(yeni_veriler)")
            
        else:
            # Mevcut en iyi parametreleri kullan
            from star_model import build_star_model, train_star_model
            from prepare_data import load_star_subset
            
            print("\n" + "="*70)
            print("EN İYİ PARAMETRELERLE YILDIZ MODELİ EĞİTİLİYOR".center(70))
            print("="*70)            # Veriyi yükle
            print("Veri yükleniyor...")
            data_path_star = 'data/star_subtypes.csv'
            X_train, X_val, X_test, y_train, y_val, y_test, le_star, scaler_star = load_star_subset(data_path_star)
            
            # Veri kontrolü - NaN/Inf değerleri tespit için
            print("Veri kontrol ediliyor...")
            nan_count_train = np.isnan(X_train).sum()
            inf_count_train = np.isinf(X_train).sum()
            if nan_count_train > 0 or inf_count_train > 0:
                print(f"UYARI: Eğitim verisinde {nan_count_train} NaN ve {inf_count_train} Inf değer bulundu.")
                print("Bu değerler otomatik olarak temizlenecek.")
                # Ekstra güvenlik - NaN'ları temizle
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
                X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
              # Etiketlerin durumunu kontrol et ve güvenli bir şekilde dönüştür
            print("\nYıldız alt tür etiketlerini hazırlıyorum...")
            # y_train doğrudan sayısal one-hot encoded format mı?
            if isinstance(y_train, np.ndarray) and len(y_train.shape) == 2:
                print("Etiketler zaten one-hot kodlanmış.")
                y_int = np.argmax(y_train, axis=1)
            # Yoksa string veya sayısal sınıf değerlerini içeren bir dizi mi?
            else:
                # Kategorik string değerleri ise LabelEncoder ile dönüştür
                if not np.issubdtype(y_train.dtype, np.number):
                    print("Kategorik etiketleri sayısal değerlere dönüştürüyorum...")
                    y_int = le_star.transform(y_train)
                else:
                    print("Etiketler zaten sayısal format.")
                    y_int = y_train
            
            # Sınıf ağırlıklarını hesapla
            unique_classes = np.unique(y_int)
            print(f"Benzersiz sınıf sayısı: {len(unique_classes)}")
            print(f"Örnek etiketler: {y_int[:5]}")
            
            # Class weights hesapla
            cw = class_weight.compute_class_weight("balanced", classes=unique_classes, y=y_int)
            cw_dict = dict(zip(unique_classes, cw))
            
            # Daha dengeli parametrelerle modeli oluştur
            print("\nYıldız modeli daha dengeli parametrelerle oluşturuluyor...")
            
            # Model boyutları
            n_features = X_train.shape[1]
            n_classes = y_train.shape[1]            # Modeli oluştur (optimize edilmiş parametrelerle)
            best_model = build_star_model(
                input_dim=n_features, 
                n_classes=n_classes,
                neurons1=490,         # Arttırıldı: 434 -> 490
                neurons2=120,         # Arttırıldı: 99 -> 120
                neurons3=120,         # Arttırıldı: 107 -> 120
                dropout1=0.4,         # Ayarlandı: 0.379 -> 0.4
                dropout2=0.35,        # Ayarlandı: 0.334 -> 0.35
                dropout3=0.25,        # Ayarlandı: 0.230 -> 0.25
                learning_rate=0.0002  # Ayarlandı: 0.00024 -> 0.0002
            )
            
            # Modeli eğit
            print("\nYıldız modeli eğitiliyor...")
            best_model, _ = train_star_model(
                best_model, 
                X_train, y_train, 
                X_val, y_val, 
                class_weights=cw_dict,
                batch_size=64,         # Daha küçük: 128 -> 64 (daha iyi genelleme)
                epochs=30,             # Daha uzun: 20 -> 30
                use_cyclic_lr=True,
                use_trending_early_stop=True
            )
            
            # Test doğruluğunu hesapla
            test_preds = best_model.predict(X_test)
            test_accuracy = (test_preds.argmax(1) == y_test.argmax(1)).mean() * 100
            print(f"\nYıldız alt türleri test doğruluğu: {test_accuracy:.2f}%")
            
            # Modeli kaydet
            best_model.save(f"outputs/optimized_star_model.keras")            
            print("\nOptimize edilmiş model kaydedildi: outputs/optimized_star_model.keras")
            print("Bu modeli kullanmak için:")
            print(">>> from tensorflow.keras.models import load_model")
            print(">>> model = load_model('outputs/optimized_star_model.keras')")
            print(">>> tahminler = model.predict(yeni_veriler)")
    except Exception as e:
        print(f"\nGelişmiş yıldız modeli eğitimi sırasında hata oluştu: {str(e)}")
        print("Ana model eğitimi başarıyla tamamlandı, gelişmiş model eğitimi adımı atlandı.")

if __name__ == '__main__':
    # Gerekli kütüphaneleri kontrol et
    try:
        print("\n--- Bağımlılıklar kontrol ediliyor ---")
        # TensorFlow zaten kontrol edildi
        
        # scikit-learn
        import sklearn
        print(f"scikit-learn sürümü: {sklearn.__version__}")
        
        # Gerekli diğer kütüphaneler
        import matplotlib
        print(f"matplotlib sürümü: {matplotlib.__version__}")
        
        import pandas
        print(f"pandas sürümü: {pandas.__version__}")
        
        # Joblib
        import joblib
        print(f"joblib sürümü: {joblib.__version__}")
        
        print("Tüm gerekli kütüphaneler mevcut.\n")
    except ImportError as e:
        print(f"Eksik kütüphane bulundu: {e}")
        print("pip install scikit-learn pandas matplotlib joblib tensorflow seaborn")
    
    # Ana modellerin eğitimi
    print("\n" + "="*70)
    print("ASTRONOMİK SINIFLANDIRICI EĞİTİMİ BAŞLATILIYOR".center(70))
    print("="*70)
      # Kullanıcıya seçenekler sun
    print("\nÇalıştırılacak modlar:")
    print("1. Temel eğitim (Galaksi/Kuasar/Yıldız sınıflandırma)")
    print("2. Gelişmiş yıldız modeli eğitimi (optimize parametreler/optimizasyon)")
    print("3. Tüm modlar (temel eğitim + gelişmiş yıldız modeli)")    
    try:
        mode = int(input("\nSeçiminiz (1/2/3) [varsayılan=3]: ") or "3")
    except ValueError:
        mode = 3
        print("Geçersiz seçim, varsayılan olarak tüm modlar çalıştırılacak.")
    
    # Seçilen moda göre çalıştır
    if mode == 1:
        main()
    elif mode == 2:
        run_advanced_star_model()
    else:
        main()
        print("\n\nAna model eğitimi tamamlandı, gelişmiş yıldız modeli eğitimine geçiliyor...\n")
        run_advanced_star_model()
    
    print("\nİşlemler tamamlandı! Sonuçlar 'outputs' klasöründe.")
