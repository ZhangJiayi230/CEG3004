# CEG3004 Project - T21

## 1. Project Overview
This project focuses on Environmental Sound Classification using Digital Signal Processing and machine learning techniques. The objective is to classify audio clips into 50 sound classes while maintaining robustness under clean, noisy, and band=limited conditions.

## 2. Objectives
The main objectives of this project are:
- Train a classifier on labeled environmental sound data
- Extract meaningful DSP-based audio features
- Classify audio clips into 50 sound classes
- Improve robustness against noise and bandwidth distortion
- Produce a reproducible pipeline that generates the required model and prediction files

## 3. Dataset
The dataset used in this project is derived from the ESC-50 environmental sound dataset. It contains:
- 2000 audio clips
- 50 sound classes
- 40 clips per class
- 5 seconds per clip where each clip is stored as a mono waveform

The evaluation setup includes three versions of each audio clip:
- **Clean**: original signal
- **Noisy**: additive noise applied
- **Band-limited**: restricted frequency content

## 4. Overall Pipeline
The implemented pipeline is:

1. Load audio
2. Apply preprocessing
3. Extract DSP-based features
4. Standardise the feature vectors
5. Train a machine learning classifier
6. Generate predictions for submission data
7. Save the trained model and prediction CSV

## 5. DSP Methodology

### 5.1 Audio Loading
Audio is loaded using `librosa.load()` with:
- Mono conversion enabled
- Target sampling rate of `16000 Hz`

Any invalid numeric values are cleaned using `np.nan_to_num()`.

### 5.2 Preprocessing
The preprocessing steps implemented are:

- **Peak amplitude normalisation**
Each waveform is scaled by its maximum amplitude.

- **Silence trimming**
Leading and trailing silence are removed using `librosa.effects.trim()` with `top_db=30`.

- **Pre-emphasis filtering**
A pre-emphasis filter is applied using coefficient `0.97` to emphasise higher frequency components.

- **Fixed-length truncation**
Each clip is adjusted to a fixed duration of **5 seconds** to ensure consistent feature dimensions.

These preprocessing steps help reduce inconsistency across audio samples before feature extraction.

### 5.3 Feature Extraction

The feature vector combines cepstral, time-frequency, and spectral descriptors.

#### A. MFCC-based features
The baseline feature block uses:
- 20 MFCC coefficients
- Delta MFCC
- Delta-delta MFCC

For each of these, the following statistics are computed across time:
- Mean
- Standard deviation
  
#### B. Log-mel spectrogram features
A Log-mel spectogram is computed using:
- `n_fft = 1024`
- `hop_length = 256`
- `n_mels = 64`

The following summary statistics are extracted from the log-mel representation:
- mean
- standard deviation
- median

These features provide additional information about the energy distribution across frequency bands.

#### C. Spectral features
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero-crossing rate

For each of these, the following statistics are computed:
- mean
- standard deviation

These features complement the MFCC and log-mel features by describing spectral shape and signal activity.

### 5.4 Robustness Strategy
Since the project evaluates performance under clean, noisy, and band-limited conditions, additional robustness-oriented improvements can include:
- Additive noise augmentation
- Gain variation
- Random bandpass or band-limiting augmentation

## 6. Model
The final classifier is implemented as a scikit-learn pipeline:

- `StandardScaler()`
- `SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced')`

The model is trained using a stratified train-validation split:
- validation size: `20%`
- random seed: `42`

The evaluation metric used during validation is **Macro-F1**, which is suitable for multi-class classification with balanced attention across all 50 classes.

## 7. Experiments and Results
Multiple feature and model configurations were tested to improve performance.

### Experiment 1: Baseline Logistic Regression
The first experiment used Logistic Regression as the baseline classifier.

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        max_iter=3000,
        class_weight='balanced',
        C=3.0,
        solver='lbfgs'
    ))
])

Validation Macro-F1: 0.5069

This result provided a useful baseline, but the performance suggested that a linear classifier was not sufficient to model the more complex class boundaries in the extracted audio feature space.

### Experiment 2: RBF SVM with C = 10
The second experiment replaced Logistic Regression with a Support Vector Machine using an RBF kernel.

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced'
    ))
])

Validation Macro-F1: 0.6236

This produced a clear improvement over the baseline, showing that a non-linear classifier was more suitable for the handcrafted DSP features.

### Experiment 3: RBF SVM with C = 5

A further experiment was performed by reducing the regularisation parameter from C=10 to C=5.

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(
        kernel='rbf',
        C=5,
        gamma='scale',
        class_weight='balanced'
    ))
])

Validation Macro-F1: 0.6407

This configuration achieved the best validation performance among the tested models and was therefore selected as the final model.

### Final Model Selection

The final selected model is the RBF SVM with C=5. Compared with the baseline Logistic Regression model, it achieved a substantially higher Macro-F1 score. It also performed better than the C=10 SVM setting, suggesting that C=5 provided a better balance between fitting the training data and generalising to unseen validation samples.

## 8. Key Observations
- MFCC-based features provide a strong baseline for ESC.
- Adding log-mel statistics improves representation of spectral energy distribution.
- Spectral descriptors provide complementary information beyond cepstral features.
- Model performance depends strongly on the interaction between feature design and classifier hyperparameters.
  
## How to Run
1. Open the notebook in Google Colab.
2. Install the required libraries.
3. Run all cells in order.
4. Train the model and generate predictions.
5. The output model file and prediction CSV will be saved automatically.
