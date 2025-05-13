import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_star_model(input_dim: int, n_classes: int, 
                    neurons1=256, neurons2=128, neurons3=64, 
                    dropout1=0.4, dropout2=0.4, dropout3=0.3,
                    learning_rate=0.001, **kwargs):
    """
    Yıldız türlerini sınıflandırmak için derin sinir ağı modeli oluşturur.
    
    Parametreler:
    - input_dim: Giriş özelliklerinin boyutu
    - n_classes: Sınıf sayısı (çıkış nöronları)
    - neurons1/2/3: Her katman için nöron sayıları
    - dropout1/2/3: Her katman için dropout oranları
    - learning_rate: Öğrenme oranı
    - **kwargs: Geriye dönük uyumluluk için ek parametreler (kullanılmaz)
    
    Returns:
    - Derlenmiş model
    """
    # Standart model (yüksek performanslı model)
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(neurons1, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout1),
        Dense(neurons2, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(), 
        Dropout(dropout2),
        Dense(neurons3, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout3),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    return model

# Hızlı eğitim için yardımcı fonksiyon
def train_star_model(model, X_train, y_train, X_val, y_val, class_weights=None, 
                    max_samples=None, batch_size=128, epochs=20, 
                    use_cyclic_lr=True, use_trending_early_stop=True):
    """
    Yıldız türü modelini gelişmiş eğitim stratejileriyle eğitmek için hızlı bir yöntem.
    
    Parametreler:
    - model: Eğitilecek model
    - X_train, y_train: Eğitim verileri
    - X_val, y_val: Doğrulama verileri
    - class_weights: Sınıf ağırlıkları (opsiyonel)
    - max_samples: Maksimum örnek sayısı (alt örnekleme için)
    - batch_size: Yığın boyutu
    - epochs: Maksimum epoch sayısı
    - use_cyclic_lr: Döngüsel öğrenme oranı kullanılsın mı
    - use_trending_early_stop: Trendi izleyen erken durdurma kullanılsın mı
    
    Returns:
    - Eğitilmiş model ve eğitim geçmişi
    """
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, Callback
    import numpy as np
    
    # Eğitim veri setini küçültme (alt örnekleme)
    if max_samples and len(X_train) > max_samples:
        print(f"Eğitim veri setini {max_samples} örneğe küçültüyorum...")
        idx = np.random.choice(len(X_train), max_samples, replace=False)
        X_train_sample = X_train[idx]
        y_train_sample = y_train[idx]
    else:
        X_train_sample, y_train_sample = X_train, y_train
    
    # Callback'leri hazırla
    callbacks = []
    
    # 1. Trendi izleyen erken durdurma
    if use_trending_early_stop:
        class TrendingEarlyStopping(Callback):
            """Uzun vadeli eğilimi izleyen erken durdurma"""
            
            def __init__(self, monitor='val_loss', patience=5, window_size=8, min_delta=0):
                super(TrendingEarlyStopping, self).__init__()
                self.monitor = monitor
                self.patience = patience
                self.window_size = window_size
                self.min_delta = min_delta
                self.wait = 0
                self.best = float('inf') if 'loss' in monitor else -float('inf')
                self.history = []
                
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                current = logs.get(self.monitor)
                if current is None:
                    return
                
                self.history.append(current)
                
                if len(self.history) >= self.window_size:
                    # Son window_size değerini kullanarak eğilimi hesapla
                    recent = self.history[-self.window_size:]
                    # Doğrusal regresyon eğimi
                    x = np.arange(len(recent))
                    slope = np.polyfit(x, recent, 1)[0]
                    
                    # 'loss' için negatif eğim iyidir, diğer metrikler için pozitif
                    is_improving = slope < -self.min_delta if 'loss' in self.monitor else slope > self.min_delta
                    
                    if not is_improving:
                        self.wait += 1
                        if self.wait >= self.patience:
                            self.model.stop_training = True
                            print(f"\nEğitim durduruldu: {self.window_size} epoch'luk eğilim iyileşmiyor.")
                    else:
                        self.wait = 0
        
        trend_stopping = TrendingEarlyStopping(
            monitor='val_categorical_accuracy',
            patience=3,
            window_size=6,
            min_delta=0.001
        )
        callbacks.append(trend_stopping)
    else:
        # Standart erken durdurma
        callbacks.append(
            EarlyStopping(
                monitor='val_categorical_accuracy',
                patience=3,
                restore_best_weights=True
            )
        )
    
    # 2. Döngüsel öğrenme oranı
    if use_cyclic_lr:
        def cyclic_learning_rate(epoch, min_lr=1e-5, max_lr=1e-3, cycle_length=5):
            """Döngüsel öğrenme oranı planı"""
            # Döngü içindeki mevcut konum (0 ile 1 arasında)
            cycle_progress = (epoch % cycle_length) / cycle_length
            
            # Kosinüs dalgası (0 ile 1 arasında, yarım döngü)
            cos_wave = 0.5 * (1 + np.cos(np.pi * (cycle_progress * 2 - 1)))
            
            # Logaritmik ölçekte yumuşak geçiş
            log_min, log_max = np.log10(min_lr), np.log10(max_lr)
            log_lr = log_min + cos_wave * (log_max - log_min)
            
            return 10 ** log_lr
        
        lr_scheduler = LearningRateScheduler(cyclic_learning_rate)
        callbacks.append(lr_scheduler)
    else:
        # Standart öğrenme oranı azaltıcı
        callbacks.append(
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        )
    
    # Eğitim
    history = model.fit(
        X_train_sample, y_train_sample,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history
