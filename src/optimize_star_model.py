import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from prepare_data import load_star_subset

def create_model(neurons1=256, neurons2=128, neurons3=64, dropout1=0.4, dropout2=0.4, 
                dropout3=0.3, learning_rate=0.001, activation='relu'):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam
    
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
    
    global n_features, n_classes
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]
    
    # KerasClassifier ile model oluştur
    model = KerasClassifier(
        build_fn=create_model,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Arama alanını tanımla
    param_grid = {
        'neurons1': [128, 256, 512],
        'neurons2': [64, 128, 256],
        'neurons3': [32, 64, 128],
        'dropout1': [0.3, 0.4, 0.5],
        'dropout2': [0.3, 0.4, 0.5],
        'dropout3': [0.2, 0.3, 0.4],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64],
        'epochs': [50, 100],
        'activation': ['relu', 'elu', 'selu']
    }
    
    # RandomizedSearchCV ile hiperparametre optimizasyonu
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # 20 farklı kombinasyon dene
        cv=3,  # 3 katlı çapraz doğrulama
        verbose=1,
        n_jobs=1  # Paralel çalışmayı kapat (Keras zaten çoklu işlem kullanıyor)
    )
    
    # Optimizasyonu başlat
    random_search.fit(X_train, y_train)
    
    # En iyi parametreleri ve skoru yazdır
    print("En iyi parametreler:", random_search.best_params_)
    print("En iyi doğruluk skoru: {:.2f}%".format(random_search.best_score_ * 100))
    
    # En iyi modeli değerlendir
    best_model = random_search.best_estimator_.model
    val_acc = best_model.evaluate(X_val, y_val)[1] * 100
    test_acc = best_model.evaluate(X_test, y_test)[1] * 100
    
    print(f"Doğrulama doğruluğu: {val_acc:.2f}%")
    print(f"Test doğruluğu: {test_acc:.2f}%")
    
    # En iyi modeli kaydet
    best_model.save('outputs/optimized_star_model.keras')