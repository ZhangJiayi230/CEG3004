# CEG3004 Project - T21

## Introduction
This project focuses on Environmental Sound Classification using Digital Signal Processing and machine learning techniques. The objective is to classify audio clips into 50 sound classes while maintaining robustness under clean, noisy, and band=limited conditions.

## Main objectives
- Train a classifier on labeled environmental sound data
- Extract meaningful DSP-based audio features
- Classify sounds into 50 classes
- Improve robustness against noise and bandwidth distortion

## Dataset
The dataset used in this project is derived from the ESC-50 environmental sound dataset. It contains 2000 audio clips across 50 classes, with 40 clips per class. Each clip is 5 seconds long and stored as a mono waveform. The evaluation set includes clean, noisy, and band-limited versions of the audio to test model robustness.

## DSP Methodology

### Preprocessing
- Audio loading and resampling
- Amplitude normalisation
- Silence trimming
- Padding or truncation to fixed length
- Optional pre-emphasis

### Feature Extraction
- MFCCs
- Log-mel spectrogram features
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff


## How to Run
1. Open the notebook in Google Colab.
2. Install the required libraries.
3. Run all cells in order.
4. Train the model and generate predictions.
5. The output model file and prediction CSV will be saved automatically.
