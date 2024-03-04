# Keyword Spotting using speech_commands_v0.02 Dataset

This repository contains code for keyword spotting using the speech_commands_v0.02 dataset. The project involves data preprocessing, feature extraction, and training models for keyword recognition. (About suggested data distribution, see: https://arxiv.org/pdf/1804.03209.pdf and link to download the data: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

## Data Preprocessing

### Enhancing Audio Quality

Several functions are provided for enhancing audio quality:

- **Remove Reverberation**: Eliminates reverberations from audio.
- **Noise Reduction with Spectral Subtraction**: Effective for reducing background noise.
- **Noise Reduction with Simple Thresholding**: Reduces sudden spikes or bursts of noise.

While these functions are available, they are not applied directly to our training data. They require extensive investigation of the audios to determine the appropriate enhancements. The provided parameters may work well for low-quality audios but can potentially worsen the quality of good audios. Further investigation and tuning are necessary to improve the results effectively.

### Feature Extraction

Two main feature extraction techniques are implemented:

- **compute_mfcc**:
    - Trimming audio.
    - Conversion to mel spectrogram.
    - Conversion to log scale (dB).
    - Normalization.
    - Calculation of Mel Frequency Cepstral Coefficients (MFCCs).
    - Calculation of delta coefficients.

- **normalize_input_size**:
    - Ensures the crop of the block of size 101 MFCCs for each input.

These feature extraction methods are crucial for preparing the audio data for training and classification tasks.

## Training Models

### CNN Model

- Early stopping occurred at the 10th epoch.
- Top-One error accuracy: 91.76%
- Precision: 92.09%
- Recall: 91.75%
- Fscore: 91.80%

### CRNN Model

- Early stopping occurred at the 20th epoch.
- Top-One error accuracy: 92.39%
- Precision: 93.03%
- Recall: 92.37%
- Fscore: 92.55%

### Autoencoder + SVM Model

- As the Autoencoder (AE) doesn't perform well with imbalanced data, the majority class is reduced using a bash script to ~balance the data.
- Cost-sensitive auto-encoders can be implemented in order to deal with the imbalance(original) data. (see more in https://www.sciencedirect.com/science/article/abs/pii/S2542660519302033)
- Applying Data Sampling techniques such as oversampling or undersampling didn't improve performance significantly in training with imbalanced data. (They have been applied only after AE training, to latent vector data for SVM)
- Applying SMOTEENN improved accuracy from ~65% to ~78% and also increased the F1 score by ~10% in the imbalanced data. However, it did not fully resolve the bias against the _unknown_ class.
- GridSearch is applied on a small proportion of the train data to find the best performing parameters for the Support Vector Machine (SVM). Below results obtained with ~balanced data.
- Top-One error accuracy: 83.16%
- Precision: 83.75%
- Recall: 83.13%
- Fscore: 83.30%

## Notebooks

- `kws_spectrograms.ipynb`: Notebook for testing spectrograms with a simple CNN model.
- `kws_project.ipynb`: Main Jupyter notebook containing the implementation of keyword spotting with the speech_commands_v0.02 dataset.
- `demo.ipynb`: Load the trained model, record an audio of 1 second with sample rate of 16k and predict the class which it belongs to.
