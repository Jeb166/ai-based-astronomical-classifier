from tensorflow.keras import Sequential, layers

def build_wide_bn(in_features: int):
    """Dense → BN → ReLU → Dropout stack."""
    return Sequential([
        layers.Dense(64,        input_shape=(in_features,), activation=None,
                     kernel_initializer="he_uniform"),
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.20),

        layers.Dense(64, activation=None, kernel_initializer="he_uniform"),
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.20),

        layers.Dense(32, activation=None, kernel_initializer="he_uniform"),
        layers.BatchNormalization(), layers.ReLU(), layers.Dropout(0.10),

        layers.Dense(16, activation=None, kernel_initializer="he_uniform"),
        layers.BatchNormalization(), layers.ReLU(),

        layers.Dense(3,  activation="softmax", name="output")
    ])
