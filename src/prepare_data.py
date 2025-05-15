import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical
from imblearn.over_sampling import SMOTE

def load_and_prepare(filename: str):
    # Read and shuffle data
    sdss_df = pd.read_csv(filename, encoding='utf-8')
    sdss_df = sdss_df.sample(frac=1)

    # Drop physically insignificant columns
    sdss_df = sdss_df.drop(
        ['objid', 'specobjid', 'run', 'rerun', 'camcol', 'field'],
        axis=1
    )

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
    test_df = sdss_df.iloc[-test_count:]

    # Extract features
    X_train = train_df.drop(['class'], axis=1)
    X_validation = validation_df.drop(['class'], axis=1)
    X_test = test_df.drop(['class'], axis=1)

    # One-hot encode labels
    le = LabelEncoder()
    le.fit(sdss_df['class'])
    encoded_Y = le.transform(sdss_df['class'])
    onehot_labels = to_categorical(encoded_Y)

    y_train = onehot_labels[:train_count]
    y_validation = onehot_labels[train_count:train_count+val_count]
    y_test = onehot_labels[-test_count:]

    # Scale features (fit on train only)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_validation = pd.DataFrame(scaler.transform(X_validation), columns=X_validation.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_validation, X_test, y_train, y_validation, y_test, sdss_df

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
    # NaN değerleri ilgili sütunların medyanları ile doldur
    star_df = star_df.fillna(star_df.median())

    # ------------------------------------------------------------------
    # 4)  Özellik / etiket ayrımı ve split
    # ------------------------------------------------------------------
    y = star_df["subClass"]
    X = star_df.drop(
        ["class", "subClass", "objid", "specobjid",
         "run", "rerun", "camcol", "field"],
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

    le = LabelEncoder().fit(y_train)
    y_train_oh = to_categorical(le.transform(y_train))
    y_val_oh   = to_categorical(le.transform(y_val))
    y_test_oh  = to_categorical(le.transform(y_test))    # SMOTE'yi yalnızca eğitim setine uygula
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
            
        smote = SMOTE(
            sampling_strategy={
                'OB': 10000,  # Nadir sınıfların sayısını artır ama aşırı artırma
                'M': 10000, 
                'WD': 20000,
                'G': 26000, 
                'K': 26000,
                'A': 26000,
            },
            k_neighbors=5,  # Nadir sınıflarda yeterli komşu olmayabilir, değeri düşürdük
            random_state=42
        )
        X_train_res, y_train_res = smote.fit_resample(X_train_df, le.transform(y_train))
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

