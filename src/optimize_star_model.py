import numpy as np
from itertools import product
import random
from prepare_data import load_star_subset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# create_model fonksiyonu aynı kalabilir
def create_model(n_features, n_classes, neurons1=256, neurons2=128, neurons3=64, 
                dropout1=0.4, dropout2=0.4, dropout3=0.3, 
                learning_rate=0.001, activation='relu'):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(neurons1, activation=activation, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(dropout1),
        Dense(neurons2, activation=activation, kernel_regularizer=l2(1e-4)),
        BatchNormalization(), 
        Dropout(dropout2),
        Dense(neurons3, activation=activation, kernel_regularizer=l2(1e-4)),
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

if __name__ == "__main__":
    # Veriyi yükle
    data_path_star = 'data/star_subtypes.csv'
    X_train, X_val, X_test, y_train, y_val, y_test, le_star, scaler_star = load_star_subset(data_path_star)
    
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]
    
    # Parametre seçeneklerini tanımla
    param_options = {
        'neurons1': [256, 512],  # Daha az seçenek
        'neurons2': [128],
        'neurons3': [64],
        'dropout1': [0.4],
        'dropout2': [0.4],
        'dropout3': [0.3],
        'learning_rate': [0.001],
        'batch_size': [32],
        'epochs': [20],
        'activation': ['relu']
    }
    
    # 20 rastgele kombinasyon oluştur
    param_keys = list(param_options.keys())
    n_iterations = 10  # 20 yerine 10 kombinasyon
    results = []
    
    # En iyi modeli takip etmek için değişkenler
    best_val_acc = 0
    best_params = None
    best_model = None
    
    print("Manuel hiperparametre optimizasyonu başlıyor...")
    print(f"Toplam {n_iterations} farklı kombinasyon denenecek")
    
    for i in range(n_iterations):
        # Rastgele bir parametre kombinasyonu seç
        params = {k: random.choice(param_options[k]) for k in param_keys}
        
        print(f"\nKombinasyon {i+1}/{n_iterations}:")
        print(params)
        
        # Modeli oluştur ve eğit
        model = create_model(
            n_features, n_classes,
            neurons1=params['neurons1'],
            neurons2=params['neurons2'],
            neurons3=params['neurons3'],
            dropout1=params['dropout1'],
            dropout2=params['dropout2'],
            dropout3=params['dropout3'],
            learning_rate=params['learning_rate'],
            activation=params['activation']
        )
        
        # Modeli eğit
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, verbose=1)
            ],
            verbose=1
        )
        
        # Doğrulama doğruluğunu hesapla
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        # Sonuçları kaydet
        results.append({
            'params': params,
            'val_acc': val_acc
        })
        
        print(f"Doğrulama doğruluğu: {val_acc*100:.2f}%")
        
        # En iyi modeli güncelle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params
            best_model = model
            print(f"Yeni en iyi model! Doğrulama doğruluğu: {val_acc*100:.2f}%")
    
    # En iyi parametreleri yazdır
    print("\n\nEn iyi parametreler:")
    print(best_params)
    print(f"En iyi doğrulama doğruluğu: {best_val_acc*100:.2f}%")
    
    # Test doğruluğunu hesapla
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print(f"Test doğruluğu: {test_acc*100:.2f}%")
    
    # En iyi modeli kaydet
    best_model.save('outputs/optimized_star_model.keras')
