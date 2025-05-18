import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import platform
import sys
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

from prepare_data import load_and_prepare
from model import build_model

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
    data_path = 'data/skyserver.csv'
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)    # ------------------------------------------------------------------
    # 1) LOAD GALAXY/QSO/STAR DATA
    # ------------------------------------------------------------------
    X_tr, X_val, X_te, y_tr, y_val, y_te, df_full, scaler = load_and_prepare(data_path)
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
    plt.show()    # ------------------------------------------------------------------
    # 6) SAVE MODELS
    # ------------------------------------------------------------------
    try:
        print("\nModeller kaydediliyor...")
        dnn.save(f"{out_dir}/dnn_model.keras")
        print(f"DNN modeli başarıyla kaydedildi: {out_dir}/dnn_model.keras")
        
        import joblib
        import time
        import platform
        from datetime import datetime
        
        # Modelleri kaydet
        joblib.dump(rf, f"{out_dir}/rf_model.joblib")
        print(f"Random Forest modeli başarıyla kaydedildi: {out_dir}/rf_model.joblib")
        
        joblib.dump(scaler, f"{out_dir}/scaler.joblib")
        print(f"Scaler başarıyla kaydedildi: {out_dir}/scaler.joblib")
        
        # Model metadatasını hazırla
        try:
            model_info = {
                'version': '1.0',
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_file': data_path,
                'feature_count': X_tr.shape[1],
                'class_counts': {cls: int((df_full['class'] == cls).sum()) for cls in labels},
                'dnn_params': {
                    'layers': [l.name for l in dnn.layers],
                    'neurons': [l.units if hasattr(l, 'units') else None for l in dnn.layers]
                },
                'rf_params': {
                    'n_estimators': rf.n_estimators,
                    'max_depth': rf.max_depth if hasattr(rf, 'max_depth') else None,
                    'oob_score': float(rf.oob_score_),
                    'random_state': rf.random_state
                },
                'ensemble_weight': float(best_w),
                'features': list(X_tr.columns),
                'accuracy': {
                    'dnn': float(dnn_acc),
                    'rf': float(rf_acc),
                    'ensemble': float(ens_acc)
                },
                'system_info': {
                    'python_version': platform.python_version(),
                    'tensorflow_version': tf.__version__,
                    'sklearn_version': sklearn.__version__ if 'sklearn' in sys.modules else None,
                    'os': platform.system(),
                    'processor': platform.processor()
                }
            }
            
            # Metadatayı kaydet
            joblib.dump(model_info, f"{out_dir}/model_metadata.joblib")
            print(f"Model metadata başarıyla kaydedildi: {out_dir}/model_metadata.joblib")
            
            # Ensemble parametrelerini ayrıca kaydet
            ensemble_params = {
                'best_w': float(best_w),
                'dnn_model_path': f"{out_dir}/dnn_model.keras",
                'rf_model_path': f"{out_dir}/rf_model.joblib",
                'scaler_path': f"{out_dir}/scaler.joblib"
            }
            joblib.dump(ensemble_params, f"{out_dir}/ensemble_params.joblib")
            print(f"Ensemble parametreleri başarıyla kaydedildi: {out_dir}/ensemble_params.joblib")
            
        except Exception as e:
            print(f"\nModel metadata kaydedilirken hata oluştu: {e}")
    
    except Exception as e:
        print(f"\nModeller kaydedilirken hata oluştu: {e}")
      # İşlem tamamlandı mesajı
    print("\nTemel sınıflandırma modeli eğitimi tamamlandı!")
    print(f"Modeller {out_dir} klasörüne kaydedildi.")
    
    # ------------------ helper for UI / further use -----------------
    global full_predict
    def full_predict(sample_array):
        """Return GALAXY/QSO/STAR classification for each input row."""
        p_ens = best_w*dnn.predict(sample_array) + (1-best_w)*rf.predict_proba(sample_array)
        primary = p_ens.argmax(1)
        out = [labels[cls] for cls in primary]
        return out

    # leave models in global scope for interactive sessions
    globals().update({'dnn': dnn, 'rf': rf, 'best_w': best_w,
                      'labels': labels})
    
    print("\n" + "="*70)
    print("MODEL EĞİTİMİ TAMAMLANDI".center(70))
    print("="*70)
    print(f"\nTüm modeller '{out_dir}' klasörüne kaydedildi.")
    print("\nÖnemli dosyalar:")
    print(f"- {out_dir}/dnn_model.keras: Ana DNN modeli")
    print(f"- {out_dir}/rf_model.joblib: Random Forest modeli")
    
    # Modelinizi kullanmak için:
    print("\nBu modeli kullanmak için:")
    print(">>> from main import full_predict")
    print(">>> sonuclar = full_predict(yeni_veriler)")

def print_helper():
    """Kaydedilen modeller hakkında bilgi verir"""
    out_dir = 'outputs'
    print("\nKaydedilen dosyalar:")
    print(f"- {out_dir}/dnn_model.keras: Derin sinir ağı modeli")
    print(f"- {out_dir}/rf_model.joblib: Random Forest modeli")
    print(f"- {out_dir}/[çeşitli görseller].png: Performans grafikleri")

def analyze_feature_importance(rf_model, feature_names):
    """RF modelinde özellik önemi analizi yapar"""
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    # Çıktı dizini
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    
    # Özellik önemlerini al
    feature_importance = rf_model.feature_importances_
    
    # Önem sırasına göre sırala
    sorted_idx = np.argsort(feature_importance)
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    
    # Tüm özelliklerin grafiği
    plt.subplot(1, 2, 1)
    plt.title("Özellik Önemliliği (Artan Sırada)")
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='skyblue')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Önem Derecesi')
    
    # En önemli 10 özelliğin grafiği
    plt.subplot(1, 2, 2)
    top_idx = sorted_idx[-10:]  # En önemli 10 özelliğin indeksleri
    plt.title("En Önemli 10 Özellik")
    plt.barh(range(len(top_idx)), feature_importance[top_idx], color='lightgreen')
    plt.yticks(range(len(top_idx)), [feature_names[i] for i in top_idx])
    plt.xlabel('Önem Derecesi')
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_importance.png", dpi=150)
    plt.show()
    
    # En önemli özellikleri çıktı olarak ver
    importance_df = pd.DataFrame({
        'Özellik': feature_names,
        'Önem': feature_importance
    }).sort_values('Önem', ascending=False)
    
    print("\nTÜM ÖZELLİKLERİN ÖNEM SIRASI:")
    pd.set_option('display.max_rows', 100)
    print(importance_df)
    pd.reset_option('display.max_rows')
    
    # En önemli 10 özellik
    print("\nEN ÖNEMLİ 10 ÖZELLİK:")
    print(importance_df.head(10))
    
    # Ra ve dec özelliklerinin önemleri
    try:
        ra_importance = importance_df[importance_df['Özellik'] == 'ra']['Önem'].values[0]
        dec_importance = importance_df[importance_df['Özellik'] == 'dec']['Önem'].values[0]
        
        ra_rank = importance_df[importance_df['Özellik'] == 'ra'].index[0] + 1
        dec_rank = importance_df[importance_df['Özellik'] == 'dec'].index[0] + 1
        
        print("\nASTRONOMİK REFERANS KOORDİNATLARI ANALİZİ:")
        print(f"- 'ra' özelliği:  Önem değeri = {ra_importance:.6f}, Sıralaması: {ra_rank}/{len(feature_names)}")
        print(f"- 'dec' özelliği: Önem değeri = {dec_importance:.6f}, Sıralaması: {dec_rank}/{len(feature_names)}")
    except:
        print("\nRa ve dec özellikleri veri setinde bulunamadı.")
    
    return importance_df

def analyze_feature_groups(importance_df):
    """Özellik gruplarının önem analizini yapar"""
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    import re
    
    # Çıktı dizini
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    
    # Özellik grupları tanımla
    feature_groups = {
        'Astronomik Koordinatlar': ['ra', 'dec'],
        'Temel Parlaklıklar': ['u', 'g', 'r', 'i', 'z'],
        'Renk İndeksleri': ['u_g', 'g_r', 'r_i', 'i_z'],
        'Spektral Özellikler': [col for col in importance_df['Özellik'] if re.match(r'plate|mjd|fiberid', col)]
    }
    
    # Diğer sütunları "Diğer" kategorisine ekle
    all_grouped_features = []
    for group in feature_groups.values():
        all_grouped_features.extend(group)
        
    feature_groups['Diğer'] = [col for col in importance_df['Özellik'] if col not in all_grouped_features]
    
    # Her grubun ortalama önemini hesapla
    group_importance = {}
    for group_name, features in feature_groups.items():
        # İlgili grup özelliklerini filtrele
        group_features = importance_df[importance_df['Özellik'].isin(features)]
        if not group_features.empty:
            avg_importance = group_features['Önem'].mean()
            group_importance[group_name] = {
                'ortalama_önem': avg_importance,
                'özellik_sayısı': len(group_features),
                'özellikler': group_features['Özellik'].tolist()
            }
    
    # Sonuçları görselleştir
    plt.figure(figsize=(12, 6))
    
    # Ortalama önem grafiği
    group_names = list(group_importance.keys())
    avg_importances = [group_importance[g]['ortalama_önem'] for g in group_names]
    feature_counts = [group_importance[g]['özellik_sayısı'] for g in group_names]
    
    # Renk haritası - özellik sayısına göre renk tonu değişsin
    colors = plt.cm.viridis(np.array(feature_counts) / max(feature_counts))
    
    # Grafik çizimi
    bars = plt.bar(group_names, avg_importances, color=colors)
    plt.title('Özellik Gruplarının Ortalama Önem Değerleri')
    plt.ylabel('Ortalama Önem')
    plt.xticks(rotation=45)
    
    # Her çubuğun üstüne özellik sayısını yaz
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                f'n={feature_counts[i]}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/feature_group_importance.png", dpi=150)
    plt.show()
    
    # Grup önemlerini yazdır
    print("\nÖZELLİK GRUPLARININ ORTALAMA ÖNEMİ:")
    for group_name, info in sorted(group_importance.items(), key=lambda x: x[1]['ortalama_önem'], reverse=True):
        print(f"\n{group_name}: {info['ortalama_önem']:.6f} (Özellik Sayısı: {info['özellik_sayısı']})")
        print(f"  İçerdiği özellikler: {', '.join(info['özellikler'])}")
        
    return group_importance

def evaluate_models():
    """Kaydedilmiş modelleri değerlendirir ve karşılaştırır"""
    import os
    import joblib
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    import tensorflow as tf
    
    # Çıktı dizini
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    
    # Verileri yükle
    print("Veriler hazırlanıyor...")
    from prepare_data import load_and_prepare
    data_path = 'data/skyserver.csv'
    X_tr, X_val, X_te, y_tr, y_val, y_te, df_full, _ = load_and_prepare(data_path)
    y_te_lbl = y_te.argmax(1)
      # Modelleri yükle
    print("Modeller yükleniyor...")
    try:
        dnn = tf.keras.models.load_model(f"{out_dir}/dnn_model.keras")
        rf = joblib.load(f"{out_dir}/rf_model.joblib")
        
        # Model metadata ve ensemble parametrelerini yüklemeyi dene
        try:
            metadata = joblib.load(f"{out_dir}/model_metadata.joblib")
            ensemble_params = joblib.load(f"{out_dir}/ensemble_params.joblib")
            
            # Metadata bilgilerini göster
            print("\n--- MODEL METADATA BİLGİLERİ ---")
            print(f"Model Sürümü: {metadata.get('version', 'Belirtilmemiş')}")
            print(f"Eğitim Tarihi: {metadata.get('training_date', 'Belirtilmemiş')}")
            print(f"Özellik Sayısı: {metadata.get('feature_count', 'Belirtilmemiş')}")
            
            # Doğruluk bilgilerini göster
            if 'accuracy' in metadata:
                print(f"\nKayıtlı Model Doğrulukları:")
                print(f"  DNN: {metadata['accuracy']['dnn']:.3f}%")
                print(f"  RF: {metadata['accuracy']['rf']:.3f}%")
                print(f"  Ensemble: {metadata['accuracy']['ensemble']:.3f}%")
            
            # Ensemble parametrelerini kullan
            best_w = ensemble_params.get('best_w', 0.5)
            print(f"Optimum DNN ağırlığı: {best_w:.2f} (metadata'dan alındı)")
            
        except (FileNotFoundError, KeyError) as e:
            print(f"Metadata yüklenirken hata: {e}")
            print("Optimum ağırlık yeniden hesaplanacak...")
            
            # En iyi ağırlığı bul (validation seti üzerinde)
            dnn_val = dnn.predict(X_val)
            rf_val = rf.predict_proba(X_val)
            y_val_lbl = y_val.argmax(1)
            
            best_w, best_acc = 0.5, 0.0
            for w in np.linspace(0.1, 0.9, 9):
                proba_acc = ((w*dnn_val+(1-w)*rf_val).argmax(1)==y_val_lbl).mean()
                if proba_acc > best_acc:
                    best_w, best_acc = w, proba_acc
            
            print(f"Optimum DNN ağırlığı: {best_w:.2f} (yeniden hesaplandı, acc={best_acc*100:.2f}%)")
    except Exception as e:
        print(f"Modeller yüklenirken hata: {e}")
        return None    # Test seti sonuçlarını hesapla
    print("\nTest sonuçları hesaplanıyor...")
    dnn_probs = dnn.predict(X_te)
    rf_probs = rf.predict_proba(X_te)
    ens_probs = best_w*dnn_probs + (1-best_w)*rf_probs
    
    dnn_preds = dnn_probs.argmax(1)
    rf_preds = rf_probs.argmax(1)
    ens_preds = ens_probs.argmax(1)
    
    dnn_acc = (dnn_preds==y_te_lbl).mean()*100
    rf_acc = (rf_preds==y_te_lbl).mean()*100
    ens_acc = (ens_preds==y_te_lbl).mean()*100
    
    print(f"\nTest Sonuçları:")
    print(f"DNN  Doğruluk: {dnn_acc:6.3f}%")
    print(f"RF   Doğruluk: {rf_acc:6.3f}%")
    print(f"ENS  Doğruluk: {ens_acc:6.3f}%")
    
    # Ensemble model için sınıflandırma raporu
    labels = np.unique(df_full['class'])
    print("\nEnsemble Model Sınıflandırma Raporu:")
    print(classification_report(y_te_lbl, ens_preds, target_names=labels))
    
    # Görselleştirme
    plt.figure(figsize=(15, 10))
    
    # Doğruluk karşılaştırması
    plt.subplot(2, 2, 1)
    model_names = ['DNN', 'Random Forest', 'Ensemble']
    accuracies = [dnn_acc, rf_acc, ens_acc]
    colors = ['skyblue', 'lightgreen', 'coral']
    
    plt.bar(model_names, accuracies, color=colors)
    plt.ylim([min(accuracies) - 1, 100])
    plt.xlabel('Model')
    plt.ylabel('Test Doğruluğu (%)')
    plt.title('Model Test Performansı Karşılaştırması')
    
    # Her modelin karışıklık matrisi
    plt.subplot(2, 2, 2)
    sns.heatmap(confusion_matrix(y_te_lbl, dnn_preds), annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('DNN - Karışıklık Matrisi')
    
    plt.subplot(2, 2, 3)
    sns.heatmap(confusion_matrix(y_te_lbl, rf_preds), annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, cmap='Greens')
    plt.title('Random Forest - Karışıklık Matrisi')
    
    plt.subplot(2, 2, 4)
    sns.heatmap(confusion_matrix(y_te_lbl, ens_preds), annot=True, fmt='d',
                xticklabels=labels, yticklabels=labels, cmap='Oranges')
    plt.title('Ensemble - Karışıklık Matrisi')
    plt.tight_layout()
    plt.savefig(f"{out_dir}/model_comparison.png", dpi=150)
    plt.show()
    
    print(f"\nModel karşılaştırma görseli '{out_dir}/model_comparison.png' olarak kaydedildi.")
    
    return dnn, rf, best_w

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
    
    # Seçenekleri göster
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        print("\n" + "="*70)
        print("MODEL DEĞERLENDİRME MODU".center(70))
        print("="*70)
        evaluate_models()
    else:
        # Ana modellerin eğitimi
        print("\n" + "="*70)
        print("ASTRONOMİK SINIFLANDIRICI EĞİTİMİ BAŞLATILIYOR".center(70))
        print("="*70)
          
        # Ana modeli çalıştır
        main()
        
        print("\nİşlemler tamamlandı! Sonuçlar 'outputs' klasöründe.")
        print("\nEğer eğitilmiş modelleri değerlendirmek isterseniz:")
        print("python src/main.py evaluate")
