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
- MFCC coefficients
- Delta MFCC
- Delta-delta MFCC
- Mean and standard deviation statistics

MFCC features capture the overall spectral envelop and are widely used in audio classification.

#### Log-mel spectrogram features
- Log-mel spectogram
- Mean, standard deviation, and median statistics

Log-mel features provide a richer time-frequency representation and helps to capture class-specific distribution patterns.

#### Spectral features
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero-crossing rate

These features provide complementary information about brightness, spread of spectral energy, frequency concentration and signal activity.

### 5.3 Robustness Strategy
Since the project evaluates performance under clean, noisy, and band-limited conditions, additional robustness-oriented improvements can include:
- Additive noise augmentation
- Gain variation
- Random bandpass or band-limiting augmentation

## 6. Model and Training Setup
The classifier used is a Support Vector Machine (SVM) with an RBF kernel inside a scikit-learn pipeline:

- `StandardScaler()`
- `SVC(kernel='rbf', C=5, gamma='scale', class_weight='balanced')`

This model was chosen as SVMs perform well on cop

## 7. Experiments and Results
Multiple feature and model configurations were tested to improve performance.


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
