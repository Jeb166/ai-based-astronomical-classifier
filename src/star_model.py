from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2

def build_star_model(input_dim: int, n_classes: int, lightweight=True):
    """
    Yıldız türlerini sınıflandırmak için derin sinir ağı modeli oluşturur.
    
    Parametreler:
    - input_dim: Giriş özelliklerinin boyutu
    - n_classes: Sınıf sayısı (çıkış nöronları)
    - lightweight: Eğer True ise, daha az parametre içeren hafif model oluşturur
    
    Returns:
    - Derlenmiş model
    """
    if lightweight:
        # Hafif model (daha hızlı eğitim)
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(n_classes, activation='softmax')
        ])
    else:
        # Orijinal model (daha yüksek kapasite)
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(), 
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(n_classes, activation='softmax')
        ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model

# Hızlı eğitim için yardımcı fonksiyon
def train_star_model(model, X_train, y_train, X_val, y_val, class_weights=None, max_samples=None):
    """
    Yıldız türü modelini eğitmek için hızlı bir yöntem.
    
    Parametreler:
    - model: Eğitilecek model
    - X_train, y_train: Eğitim verileri
    - X_val, y_val: Doğrulama verileri
    - class_weights: Sınıf ağırlıkları (opsiyonel)
    - max_samples: Maksimum örnek sayısı (alt örnekleme için)
    
    Returns:
    - Eğitilmiş model ve eğitim geçmişi
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import numpy as np
    
    # Eğitim veri setini küçültmek için alt örnekleme (isteğe bağlı)
    if max_samples and len(X_train) > max_samples:
        print(f"Eğitim veri setini {max_samples} örneğe küçültüyorum...")
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train_sample = X_train[idx]
        y_train_sample = y_train[idx]
    else:
        X_train_sample, y_train_sample = X_train, y_train
    
    # Eğitim
    history = model.fit(
        X_train_sample, y_train_sample,
        epochs=20,  # Daha az epoch sayısı
        batch_size=128,  # Daha büyük batch size
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor='val_categorical_accuracy', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
        ],
        verbose=1
    )
    
    return model, history
