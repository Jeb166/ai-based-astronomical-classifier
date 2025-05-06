import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from prepare_data import load_and_prepare
from model import build_model


def main():
    # Path to original CSV in Colab environment
    filename = '/data/skyserver.csv'

    # Load and prepare data
    X_train, X_validation, X_test, y_train, y_validation, y_test, sdss_df = load_and_prepare(filename)

    # (Optional) Apply PCA & plot distributions or projections as in notebook
    # ... (insert here if needed exactly like notebook)

    # Build and compile DNN
    dnn = build_model(X_train.shape[1], y_train.shape[1])

    # Train DNN
    my_epochs = 50
    history = dnn.fit(
        X_train, y_train,
        epochs=my_epochs,
        batch_size=50,
        validation_data=(X_validation, y_validation)
    )

    # Evaluate on test set
    preds = pd.DataFrame(dnn.predict(X_test)).idxmax(axis=1)
    y_test_labels = y_test.dot([0, 1, 2])
    model_acc = (preds == y_test_labels).sum().astype(float) / len(preds) * 100
    print('Deep Neural Network')
    print(f'Test Accuracy: {model_acc:3.5f}')

    # Plot confusion matrix
    labels = np.unique(sdss_df['class'])
    ax = plt.subplot(1, 1, 1)
    sns.heatmap(
        confusion_matrix(y_test_labels, preds),
        annot=True, fmt='d',
        xticklabels=labels, yticklabels=labels
    )
    plt.title('Confusion Matrix')
    plt.show()

    # Plot training vs validation accuracy
    epochs_arr = range(1, my_epochs + 1)
    plt.plot(epochs_arr, history.history['categorical_accuracy'], 'r-', label='training accuracy')
    plt.plot(epochs_arr, history.history['val_categorical_accuracy'], 'b-', label='validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy')
    plt.legend()
    plt.show()

    # Validation set performance
    preds_val = pd.DataFrame(dnn.predict(X_validation)).idxmax(axis=1)
    y_val_labels = y_validation.dot([0, 1, 2])
    val_acc = (preds_val == y_val_labels).sum().astype(float) / len(preds_val) * 100
    print('Deep Neural Network')
    print(f'Validation Accuracy: {val_acc:3.5f}')

if __name__ == '__main__':
    main()
