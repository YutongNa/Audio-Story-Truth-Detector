# Audio-Story-Truth-Detector
This project investigates the task of identifying whether a narrated story is truthful or deceptive based solely on its audio recording. Leveraging machine learning techniques, we extract acoustic and prosodic features from 30-second spoken stories and use them to train a variety of classification models. The goal is to analyze speech patterns that may signal deception—such as tone, pitch, and rhythm—and evaluate how effectively these features can predict the veracity of a story without relying on textual content.

## Techniques

This project applies a complete machine learning pipeline to the task of audio-based deception detection.  Key steps and techniques include:


* **Audio Preprocessing**:

  * Conversion and normalization using `pydub` and `librosa`.
  * Extraction of acoustic features such as MFCCs, chroma, spectral contrast, and zero-crossing rate.

* **Data Balancing**:

  * Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to handle class imbalance between truthful and deceptive stories.

* **Feature Engineering and Scaling**:

  * Standardization of features using `StandardScaler`.
  * Integration of categorical attributes via `OneHotEncoder`.

* **Model Training and Selection**:

  * Evaluated multiple classifiers including:

    * **Support Vector Machine (SVM)**
    * **Random Forest**
    * **LightGBM (Gradient Boosting)**
  * Used `GridSearchCV` for hyperparameter tuning.

* **Evaluation Metrics**:

  * Accuracy, confusion matrix, classification report
  * ROC curve and AUC score

## Core Packages Used

* `librosa` – audio feature extraction
* `pydub` – audio file manipulation
* `scikit-learn` – preprocessing, modeling, and evaluation
* `lightgbm` – efficient gradient boosting
* `imblearn` – handling class imbalance (SMOTE)
* `matplotlib`, `seaborn` – data visualization
* `pandas`, `numpy` – data handling and processing
* `tqdm` – progress tracking

## Dataset Description

The dataset consists of **100 audio samples** with the following components:

* **Audio Recordings**: 30-second narrated stories from various speakers
* **Language Attribute**: Language used in each recording
* **Story Type Attribute**: Binary classification indicating whether the story is true or false

**Note**: Due to file size limitations and privacy considerations, the original audio files are not included in this repository.

