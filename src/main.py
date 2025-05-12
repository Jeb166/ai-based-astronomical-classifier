# main.py — DNN + RF ensemble + STAR subtype model (Tüm optimizasyonları da içerecek şekilde)

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    joblib.dump(rf, f"{out_dir}/rf_model.joblib")    # ------------------------------------------------------------------
    # 7) STAR SUB‑CLASS MODEL
    # ------------------------------------------------------------------
    
    # A) Yıldız veri setini yükle
    print("\n" + "="*70)
    print("YILDIZ ALT TÜR MODELİ EĞİTİMİ VE OPTİMİZASYONU".center(70))
    print("="*70)
    
    Xs_tr, Xs_val, Xs_te, ys_tr, ys_val, ys_te, le_star, scaler_star = load_star_subset(data_path_star)
    
    # Sınıf ağırlıklarını hesapla
    y_int = ys_tr.argmax(1)
    cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_int), y=y_int)
    cw_dict = dict(enumerate(cw))
      # B) Farklı model mimarilerini karşılaştır
    print("\nFARKLI MODEL MİMARİLERİNİ KARŞILAŞTIRMA")
    print("-" * 40)
    
    # Model boyutları
    n_features = Xs_tr.shape[1]
    n_classes = ys_tr.shape[1]
    
    # Test edilecek model türleri
    model_types = [
        ("Standart", "standard"),
        ("Hafif", "lightweight"),
        ("Ayrılabilir", "separable"),
        ("Ağaç Yapılı", "tree")
    ]
    
    # Her modeli test et ve sonuçları sakla
    results = []
    max_samples_for_test = 20000  # Test için örnek sınırla
    
    print(f"Model karşılaştırması için {max_samples_for_test} örnek kullanılacak...")
    print("Not: Bu adım sadece model seçimi içindir. Final model tüm veri kullanılarak eğitilecektir.\n")
    
    for model_name, model_type in model_types:
        print(f"\n{model_name} model test ediliyor...")
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
            max_samples=max_samples_for_test,
            batch_size=128,
            epochs=10,  # Hızlı test için az epoch
            use_cyclic_lr=True,
            use_trending_early_stop=True
        )
        
        # Eğitim süresini ve test doğruluğunu hesapla
        training_time = time.time() - start_time
        test_acc = (model.predict(Xs_te).argmax(1) == ys_te.argmax(1)).mean() * 100
        
        # Sonuçları yazdır
        print(f"{model_name} model eğitim süresi: {training_time:.2f} saniye")
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
    print("\nMODEL KARŞILAŞTIRMASI SONUÇLARI")
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
          f"({fastest_model['time']:.2f} saniye)")    # C) Seçilen modeli tam veri üzerinde eğit
    print("\nSEÇİLEN MODELİ GERÇEK VERİLERDE EĞİTME")
    print("-" * 40)
    
    # En iyi doğruluk/hız dengesini bul
    for result in results:
        result['speed_score'] = (result['accuracy'] / 
                                (result['time'] / min(r['time'] for r in results)))
    
    # En iyi modeli seç (doğruluk öncelikli) - Standart modeli seçiyoruz
    best_model = next((r for r in results if r['type'] == 'standard'), results[0])
    print(f"Seçilen model: {best_model['name']} "
          f"(Doğruluk: {best_model['accuracy']:.2f}%, "
          f"Hız skoru: {best_model['speed_score']:.2f})")
    
    # Ana modeli oluştur ve eğit
    print(f"\nStandart model tam veri üzerinde eğitiliyor...")
    star_net = build_star_model(
        n_features, n_classes,
        model_type='standard',
        rank=16,
        neurons1=256,
        neurons2=128,
        dropout1=0.3,
        dropout2=0.3,
        learning_rate=0.001
    )
    
    # Tam veri üzerinde eğit
    star_net, history = train_star_model(
        star_net, 
        Xs_tr, ys_tr,
        Xs_val, ys_val,
        class_weights=cw_dict,
        max_samples=None,  # Tüm verileri kullan
        batch_size=128,
        epochs=20,
        use_cyclic_lr=True,
        use_trending_early_stop=True
    )
    
    # Test doğruluğunu değerlendir
    star_acc = (star_net.predict(Xs_te).argmax(1)==ys_te.argmax(1)).mean()*100
    print(f"\nSTAR subtype Test Acc: {star_acc:.2f}%")
    
    # Modeli kaydet
    star_net.save(f"{out_dir}/star_model.keras")
    joblib.dump(le_star, f"{out_dir}/star_label_enc.joblib")
    joblib.dump(scaler_star, f"{out_dir}/star_scaler.joblib")
    
    # Tüm modelleri de kaydet (opsiyonel)
    for result in results:
        model_path = f"{out_dir}/{result['type']}_star_model.keras"
        result['model'].save(model_path)
        print(f"{result['name']} model kaydedildi: {model_path}")    # ------------------ helper for UI / further use -----------------
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
    print(f"- {out_dir}/star_model.keras: Seçilen yıldız alt tür modeli")
    for result in results:
        print(f"- {out_dir}/{result['type']}_star_model.keras: {result['name']} yıldız modeli")
    
    # Modelinizi kullanmak için:
    print("\nBu modeli kullanmak için:")
    print(">>> from main import full_predict")
    print(">>> sonuclar = full_predict(yeni_veriler)")

def run_bayesian_optimization():
    """Bayesian optimizasyonu çalıştır (ayrı bir işlev olarak)"""
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
        
        from bayesian_optimize_star import optimize_star_model_bayesian
        
        print("\n" + "="*70)
        print("BAYESIAN OPTİMİZASYON BAŞLATILIYOR".center(70))
        print("="*70)
        
        # Bayesian optimizasyonu çalıştır - Sadece standart model için
        print("\nStandart model için Bayesian optimizasyon çalıştırılıyor...")
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
        
    except Exception as e:
        print(f"\nBayesian optimizasyon çalıştırılırken hata oluştu: {str(e)}")
        print("Ana model eğitimi başarıyla tamamlandı, optimizasyon adımı atlandı.")

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
    print("1. Standart eğitim (main)")
    print("2. Bayesian hiperparametre optimizasyonu")
    print("3. Tüm modlar (eğitim + optimizasyon)")
    
    try:
        mode = int(input("\nSeçiminiz (1/2/3) [varsayılan=3]: ") or "3")
    except ValueError:
        mode = 3
        print("Geçersiz seçim, varsayılan olarak tüm modlar çalıştırılacak.")
    
    # Seçilen moda göre çalıştır
    if mode == 1:
        main()
    elif mode == 2:
        run_bayesian_optimization()
    else:
        main()
        run_bayesian_optimization()
    
    print("\nİşlemler tamamlandı! Sonuçlar 'outputs' klasöründe.")
