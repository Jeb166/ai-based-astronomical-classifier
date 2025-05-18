import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical

def load_and_prepare(filename: str):
    # Read and shuffle data
    sdss_df = pd.read_csv(filename, encoding='utf-8')
    sdss_df = sdss_df.sample(frac=1)
    
    # Kategorik sütunları tespit et (sadece 'class' dışında)
    categorical_cols = [col for col in sdss_df.select_dtypes(include=['object']).columns 
                      if col != 'class']
    
    if len(categorical_cols) > 0:
        print(f"Veri setinde kategorik sütunlar bulundu: {categorical_cols}")
        print("Bu sütunlar kaldırılıyor...")
        sdss_df = sdss_df.drop(columns=categorical_cols)    # Drop physically insignificant columns
    drop_cols = []
    for col in ['objid', 'specobjid', 'run', 'rerun', 'camcol', 'field', 'ra', 'dec']:  # ra ve dec eklendi
        if col in sdss_df.columns:
            drop_cols.append(col)
    
    if drop_cols:
        print(f"Önemli olmayan ve koordinat sütunları kaldırılıyor: {drop_cols}")
        sdss_df = sdss_df.drop(drop_cols, axis=1)

    # --- Color indices (fotometrik farklar) ---
    sdss_df["u_g"] = sdss_df["u"] - sdss_df["g"]
    sdss_df["g_r"] = sdss_df["g"] - sdss_df["r"]
    sdss_df["r_i"] = sdss_df["r"] - sdss_df["i"]
    sdss_df["i_z"] = sdss_df["i"] - sdss_df["z"]


    # Partition SDSS data (60% train, 20% validation, 20% test)
    train_count = 60000
    val_count = 20000
    test_count = 20000

    train_df = sdss_df.iloc[:train_count]
    validation_df = sdss_df.iloc[train_count:train_count+val_count]
    test_df = sdss_df.iloc[-test_count:]    # Extract features
    X_train = train_df.drop(['class'], axis=1)
    X_validation = validation_df.drop(['class'], axis=1)
    X_test = test_df.drop(['class'], axis=1)
    
    # Sayısal olmayan sütunları kontrol et ve temizle
    non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"Uyarı: Sayısal olmayan veri sütunları bulundu: {non_numeric_cols}")
        print("Bu sütunlar özellik setinden çıkarılıyor...")
        X_train = X_train.drop(columns=non_numeric_cols)
        X_validation = X_validation.drop(columns=non_numeric_cols)
        X_test = X_test.drop(columns=non_numeric_cols)

    # One-hot encode labels
    le = LabelEncoder()
    le.fit(sdss_df['class'])
    encoded_Y = le.transform(sdss_df['class'])
    onehot_labels = to_categorical(encoded_Y)
    
    # Sınıf dağılımını göster
    print("Sınıf dağılımı:")
    for i, cls in enumerate(le.classes_):
        count = (sdss_df['class'] == cls).sum()
        print(f"  {cls}: {count} örnek ({count/len(sdss_df)*100:.2f}%)")

    y_train = onehot_labels[:train_count]
    y_validation = onehot_labels[train_count:train_count+val_count]
    y_test = onehot_labels[-test_count:]    # NaN ve sonsuzluk kontrolü
    print("\nÖlçeklendirmeden önce veri kontrol ediliyor...")
    for df_name, df in zip(["X_train", "X_validation", "X_test"], [X_train, X_validation, X_test]):
        nan_count = df.isna().sum().sum()
        inf_count = ((df == np.inf) | (df == -np.inf)).sum().sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  {df_name}: {nan_count} NaN değer, {inf_count} sonsuz değer")
            # NaN ve sonsuzlukları medyan ile değiştir
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            medians = df.median()
            df.fillna(medians, inplace=True)
    
    # Scale features (fit on train only)
    print("\nVeri ölçeklendiriliyor...")
    scaler = StandardScaler()
    scaler.fit(X_train)    
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    print(f"\nÖzellik vektörü boyutu: {X_train.shape[1]} özellik")
    print(f"İlk 5 özellik: {X_train.columns[:5].tolist()}")
    print(f"Son 5 özellik: {X_train.columns[-5:].tolist() if len(X_train.columns) >= 5 else X_train.columns.tolist()}")

    return X_train, X_validation, X_test, y_train, y_validation, y_test, sdss_df, scaler
"""
from imblearn.over_sampling import SMOTE
def load_star_subset(filename: str):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from keras.utils import to_categorical

    # ------------------------------------------------------------------
    # 0) CSV oku  – sadece STAR kayıtları
    # ------------------------------------------------------------------
    df = pd.read_csv(filename, encoding="utf-8")
    star_df = df[df["class"] == "STAR"].copy()
    star_df = star_df.dropna(subset=["subClass"])

    # ------------------------------------------------------------------
    # 1)  Alt‑türü 7 ana gruba indir (OB, A, F, G, K, M, WD)
    # ------------------------------------------------------------------
    def coarse_sub(sc: str) -> str:
        sc = sc.upper()
        if sc.startswith(("O", "B")):   return "OB"
        if sc.startswith("A"):          return "A"
        if sc.startswith("F"):          return "F"
        if sc.startswith("G"):          return "G"
        if sc.startswith("K"):          return "K"
        if sc.startswith("M"):          return "M"
        return "WD"                     # beyaz‑cüce ve diğerleri

    star_df["subClass"] = star_df["subClass"].apply(coarse_sub)

    # ------------------------------------------------------------------
    # 2)  EN AZ 100 örneği olan sınıfları tut
    # ------------------------------------------------------------------
    cnt = star_df["subClass"].value_counts()
    star_df = star_df[star_df["subClass"].isin(cnt[cnt >= 100].index)]
    print("Kalan alt‑türler:\n", star_df["subClass"].value_counts())

    # ------------------------------------------------------------------
    # 3)  Renk indeksleri ekle (u‑g, g‑r, r‑i, i‑z)
    # ------------------------------------------------------------------
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
        star_df[f"{a}_{b}"] = star_df[a] - star_df[b]
      # 3.5) GELİŞMİŞ ÖZELLİK MÜHENDİSLİĞİ
    # Renk oranları (astronomide önemli)
    star_df['u_over_g'] = star_df['u'] / star_df['g']
    star_df['g_over_r'] = star_df['g'] / star_df['r']
    star_df['r_over_i'] = star_df['r'] / star_df['i']
    star_df['i_over_z'] = star_df['i'] / star_df['z']
    
    # Polinom özellikler (ikinci dereceden etkileşimler)
    star_df['u_g_squared'] = star_df['u_g'] ** 2
    star_df['g_r_squared'] = star_df['g_r'] ** 2
    star_df['r_i_squared'] = star_df['r_i'] ** 2
    star_df['i_z_squared'] = star_df['i_z'] ** 2
    
    # Tüm ikili değişkenlerin birleştirilmesi (renk-renk diyagramları)
    for i, col1 in enumerate(['u', 'g', 'r', 'i', 'z']):
        for col2 in ['u', 'g', 'r', 'i', 'z'][i+1:]:
            if col1 != col2:
                star_df[f'{col1}_mul_{col2}'] = star_df[col1] * star_df[col2]
      # Spektral indeksler (astronomik özellikler için)
    # NaN değerleri önlemek için sıfıra bölünme ve sonsuz değerler için güvenlik önlemleri
    # Redshift değerlerini sıfırdan koruma (clip fonksiyonu ile)
    star_df['redshift'] = star_df['redshift'].clip(0.001)
    # g_r sıfıra çok yakınsa sorun olabilir, bunun için de bir önlem
    star_df['g_r'] = star_df['g_r'].replace(0, 0.001)
    
    # Spektral indeksler hesaplanırken NaN kontrolü
    star_df['balhc'] = star_df['redshift'] * (star_df['u_g'] / star_df['g_r'].clip(0.001))
    star_df['caii_k'] = (star_df['u'] * star_df['g']) / star_df['r'].clip(0.001)
    star_df['mgb'] = star_df['g'] * star_df['g_r'] / star_df['redshift']
    star_df['nad'] = star_df['r'] * star_df['r_i'] / star_df['redshift']
      # Hesaplamalar sonrası oluşan NaN ve sonsuz değerleri temizle
    star_df = star_df.replace([np.inf, -np.inf], np.nan)
    
    # Sayısal ve sayısal olmayan sütunları ayır
    numeric_cols = star_df.select_dtypes(include=['number']).columns
    # Sadece sayısal sütunlar için medyan hesapla
    medians = star_df[numeric_cols].median()
    # NaN değerleri sadece sayısal sütunlar için ilgili medyanlarla doldur
    star_df[numeric_cols] = star_df[numeric_cols].fillna(medians)    # ------------------------------------------------------------------
    # 4)  Özellik / etiket ayrımı ve split
    # ------------------------------------------------------------------
    y = star_df["subClass"]
    X = star_df.drop(
        ["class", "subClass", "objid", "specobjid",
         "run", "rerun", "camcol", "field", "ra", "dec"],  # ra ve dec eklendi
        axis=1
    )

    #   70 % train  – 30 % geçici (val+test)  (stratify=y)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    #   15 % val – 15 % test  (stratify KAPALI → “1 örnek” hatası yok)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=None
    )    # ------------------------------------------------------------------
    # 5)  Ölçekle & one‑hot
    # ------------------------------------------------------------------
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    
    # Ölçeklendirme sonrası NaN değerleri kontrol et ve düzelt
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    # Etiketi fit ve dönüştür
    le = LabelEncoder().fit(y.unique())  # Tüm benzersiz etiketlerle fit et
    print(f"LabelEncoder sınıfları: {le.classes_}")
      # Dönüştürme ve one-hot encoding işlemleri
    y_train_encoded = le.transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    print(f"y_train ilk 5 değer (dönüşüm sonrası): {y_train_encoded[:5]}")
    
    y_train_oh = to_categorical(y_train_encoded)
    y_val_oh = to_categorical(y_val_encoded)
    y_test_oh = to_categorical(y_test_encoded)

    # SMOTE'yi yalnızca eğitim setine uygula
    try:
        # Son bir kez daha NaN kontrolü
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("UYARI: SMOTE öncesi verilerinizde hala NaN veya sonsuz değerler var. Temizleniyor...")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # SMOTE için X_train DataFrame'e dönüştürülmeli
        if not isinstance(X_train, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train)
        else:
            X_train_df = X_train
            
        # Önce sınıf sayılarını kontrol et
        unique_classes = np.unique(y_train)
        class_counts = pd.Series(y_train).value_counts()
        print("Eğitim veri setindeki sınıf sayıları:")
        for cls in unique_classes:
            print(f"  - {cls}: {class_counts.get(cls, 0)}")
        
        # Mevcut sınıf dağılımına göre hedef sayıları belirle
        # F sınıfını limitle, diğerlerini dengeli şekilde artır
        target_counts = {}
        for cls in unique_classes:
            if cls == 'F':
                target_counts[cls] = min(class_counts.get(cls, 0), 25000)  # F sınıfını sınırla
            elif class_counts.get(cls, 0) > 0:  # Sınıf mevcutsa
                # Sınıfın en az 15.000 örneği olsun, ancak mevcut sayıdan az olmasın
                target_counts[cls] = max(class_counts.get(cls, 0), 15000)
        
        print("SMOTE hedef sınıf sayıları:", target_counts)
        
        # Komşu sayısını sınıf boyutuna göre ayarla
        min_samples = min([count for count in class_counts.values() if count > 0])
        k_neighbors = min(5, max(1, min_samples - 1))  # Küçük sınıflarda k değerini düşür
        
        smote = SMOTE(
            sampling_strategy=target_counts,
            k_neighbors=k_neighbors,
            random_state=42
        )
        
        # SMOTE'u orijinal string etiketlerle kullan
        X_train_res, y_train_res_str = smote.fit_resample(X_train_df, y_train)
        
        # Sonuçta elde edilen etiketleri dönüştür
        y_train_res = le.transform(y_train_res_str)
        y_train_res_oh = to_categorical(y_train_res)
        
        print(f"SMOTE sonrası sınıf dağılımı: {np.bincount(y_train_res)}")
        
    except Exception as e:
        print(f"SMOTE hatası: {e}")
        print("SMOTE bypass ediliyor ve orijinal veriler kullanılıyor...")
        X_train_res = X_train
        y_train_res = le.transform(y_train)
        y_train_res_oh = to_categorical(y_train_res)

    return (X_train_res, X_val, X_test,
            y_train_res_oh, y_val_oh, y_test_oh,
            le, scaler)
"""
