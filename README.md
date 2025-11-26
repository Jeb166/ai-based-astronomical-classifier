# 🔭 AI-based Astronomical Classifier

This project implements a machine learning pipeline to classify **Galaxies**, **QSOs (Quasars)**, and **Stars** using photometric data from the **Sloan Digital Sky Survey (SDSS)**.

It demonstrates how classical machine learning techniques can be applied to real scientific datasets, combining data preprocessing, feature engineering, and predictive modeling into a reproducible workflow.

The final Random Forest model achieves approximately **86% test accuracy**, which reflects realistic performance for photometric-only SDSS classification tasks.


## 1. Project Overview

This project builds an end-to-end pipeline for astronomical object classification.
The system includes:

* Data loading and preprocessing
* Feature engineering from photometric magnitudes
* Model training using classical ML algorithms
* Evaluation on a held-out test set
* A Streamlit interface for real-time predictions

The main goal is to automate the classification of celestial objects using efficient and interpretable machine learning methods.


## 2. Motivation

Astronomical surveys, including SDSS, produce massive amounts of observational data. Manual classification is no longer feasible due to scale.

Automated classification enables:

* Faster and consistent labeling
* Detection of rare or ambiguous objects
* Scalable astronomical catalog generation
* Reduced human workload

This project shows that structured photometric data alone can be leveraged to build a reliable multi-class classifier using classical ML techniques.


## 3. Dataset

### Source

The dataset is derived from **SDSS DR17** photometric measurements and includes:

* Magnitudes: **u, g, r, i, z**
* Derived color indices (e.g., **u–g**, **g–r**, **r–i**)
* Additional numeric features

### Class Labels

* **Galaxy**
* **QSO (Quasar)**
* **Star**

### Preprocessing Steps

* Removing missing or invalid entries
* Selecting relevant photometric attributes
* Creating color indices
* Feature scaling
* Label encoding
* Stratified train–test split


## Dataset Download & Placement

### Download

Dataset files can be downloaded from:

[https://drive.google.com/drive/folders/1I4WRYGt0J2rQfdmVrJaCZiiGzDswaEc_?usp=sharing](https://drive.google.com/drive/folders/1I4WRYGt0J2rQfdmVrJaCZiiGzDswaEc_?usp=sharing)

Files include:

* `skyserver.csv` — main training dataset
* `skyserver_test_data.csv` — held-out test dataset

### Placement

Place them inside the project’s `data/` directory:

```
ai-based-astronomical-classifier/
│
├── data/
│   ├── skyserver.csv
│   └── skyserver_test_data.csv
│
├── src/
├── outputs/
└── README.md
```

The scripts automatically load data from this folder.


## 4. Methodology

### 1. Data Cleaning

Filtering invalid rows, removing irrelevant columns, and ensuring consistency.

### 2. Feature Engineering

Constructing physically meaningful color indices and selecting the strongest photometric predictors.

### 3. Scaling & Normalization

Applying appropriate feature scaling where needed.

### 4. Model Training

Models tested:

* **Random Forest** (final model)
* Fully connected **Deep Neural Network**
* Simple ensemble combinations

Random Forest was chosen due to its stability and balanced performance.

### 5. Evaluation

All experiments were run on a held-out test set.

**Final test accuracy (Random Forest): ~86%**

### 6. User Interface

A **Streamlit** interface allows manual input of photometric values for instant predictions.


## 5. Installation & Usage

### Clone the repository

```
git clone https://github.com/Jeb166/ai-based-astronomical-classifier.git
cd ai-based-astronomical-classifier
```

### Install dependencies

```
pip install -r requirements.txt
```

### Train the model

```
python src/main.py
```

### Run the Streamlit interface

```
streamlit run src/streamlit.py
```

---

## 6. Dependencies

The project uses the following Python packages:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* streamlit

Install all dependencies with:

```
pip install -r requirements.txt
```


## 7. Project Structure

```
ai-based-astronomical-classifier/
│
├── src/
│   ├── main.py            # Training script
│   ├── streamlit.py       # Web UI
│   ├── prepare_data.py    # Data preprocessing
│   ├── prediction.py      # Inference utilities
│   └── data_analysis.py   # Optional analysis tools
│
├── data/                  # Dataset files
├── outputs/               # Trained model artifacts
├── requirements.txt
└── README.md
```

## 8. Features Used

* u, g, r, i, z magnitudes
* Derived color indices
* Cleaned & normalized numeric attributes


## 9. Limitations

* Photometric-only classification
* Accuracy varies depending on dataset version
* Additional tuning may improve results
* No spectroscopic features included


## 10. Future Work

* Integrating spectroscopic measurements
* Trying XGBoost / LightGBM
* Adding SHAP explainability
* Deploying a public inference API
* Enhanced preprocessing and feature engineering


## 11. License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.


## 12. Contact

For inquiries: **[emrebas02@hotmail.com](mailto:emrebas02@hotmail.com)**
