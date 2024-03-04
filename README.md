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

The CNN model architecture:

Layer (type)                Output Shape              Param #   
=================================================================
input_1 (InputLayer)        [(None, 101, 40, 1)]      0         
                                                                 
conv2_0 (Conv2D)            (None, 82, 33, 32)        5152      
                                                                 
max_pooling2d               (None, 41, 16, 32)        0         
                                                                 
batch_normalization         (None, 41, 16, 32)        128       
                                                                 
conv2_1 (Conv2D)            (None, 32, 11, 64)        122944    
                                                                 
max_pooling2d_1             (None, 16, 5, 64)         0         
                                                                 
batch_normalization_1       (None, 16, 5, 64)         256       
                                                                 
conv2_2 (Conv2D)            (None, 12, 3, 128)        123008    
                                                                 
max_pooling2d_2             (None, 6, 1, 128)         0         
                                                                 
batch_normalization_2       (None, 6, 1, 128)         512       
                                                                 
flatten (Flatten)           (None, 768)               0         
                                                                 
dropout (Dropout)           (None, 768)               0         
                                                                 
final_fc_sigmoid (Dense)    (None, 12)                9228 


- Early stopping occurred at the 10th epoch.
- Top-One error accuracy: 91.76%
- Precision: 92.09%
- Recall: 91.75%
- Fscore: 91.80%

### CRNN Model

The CRNN model architecture:

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_19 (InputLayer)       [(None, 101, 40)]         0         
                                                                 
 conv0 (Conv1D)              (None, 82, 32)            25632     
                                                                 
 max_pooling1d (MaxPooling1  (None, 41, 32)            0         
 D)                                                              
                                                                 
 batch_normalization (Batch  (None, 41, 32)            128       
 Normalization)                                                  
                                                                 
 conv1 (Conv1D)              (None, 32, 64)            20544     
                                                                 
 max_pooling1d_1 (MaxPoolin  (None, 16, 64)            0         
 g1D)                                                            
                                                                 
 batch_normalization_1 (Bat  (None, 16, 64)            256       
 chNormalization)                                                
                                                                 
 conv2 (Conv1D)              (None, 12, 128)           41088     
                                                                 
 max_pooling1d_2 (MaxPoolin  (None, 6, 128)            0         
 g1D)                                                            
                                                                 
 batch_normalization_2 (Bat  (None, 6, 128)            512       
 chNormalization)                                                
                                                                 
 conv3 (Conv1D)              (None, 4, 256)            98560     
                                                                 
 max_pooling1d_3 (MaxPoolin  (None, 2, 256)            0         
 g1D)                                                            
                                                                 
 batch_normalization_3 (Bat  (None, 2, 256)            1024      
 chNormalization)                                                
                                                                 
 gru0 (GRU)                  (None, 2, 128)            148224    
                                                                 
 dropout (Dropout)           (None, 2, 128)            0         
                                                                 
 flatten_9 (Flatten)         (None, 256)               0         
                                                                 
 fc (Dense)                  (None, 12)                3084 



- Early stopping occurred at the 20th epoch.
- Top-One error accuracy: 92.39%
- Precision: 93.03%
- Recall: 92.37%
- Fscore: 92.55%

### Autoencoder + SVM Model

The Autoencoder (AE) + SVM model architecture:

ENCODER:
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_109 (Conv2D)         (None, 101, 40, 32)       320       
                                                                 
 max_pooling2d_109 (MaxPool  (None, 51, 20, 32)        0         
 ing2D)                                                          
                                                                 
 conv2d_110 (Conv2D)         (None, 51, 20, 64)        18496     
                                                                 
 max_pooling2d_110 (MaxPool  (None, 26, 10, 64)        0         
 ing2D)                                                          
                                                                 
 flatten_44 (Flatten)        (None, 16640)             0         
                                                                 
 dropout_16 (Dropout)        (None, 16640)             0         
                                                                 
 dense_88 (Dense)            (None, 128)               2130048   
                                                                 
=================================================================


DECODER:
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_89 (Dense)            (None, 33280)             4293120   
                                                                 
 reshape_44 (Reshape)        (None, 26, 10, 128)       0         
                                                                 
 conv2d_transpose_153 (Conv  (None, 52, 20, 64)        73792     
 2DTranspose)                                                    
                                                                 
 cropping2d_76 (Cropping2D)  (None, 51, 20, 64)        0         
                                                                 
 conv2d_transpose_154 (Conv  (None, 102, 40, 32)       18464     
 2DTranspose)                                                    
                                                                 
 cropping2d_77 (Cropping2D)  (None, 101, 40, 32)       0         
                                                                 
 conv2d_transpose_155 (Conv  (None, 101, 40, 1)        289       
 2DTranspose)                                                    
                                                                 
=================================================================


=================================================================
SVC(C=1, class_weight='balanced', gamma=0.001)
=================================================================


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

- `spectrogram_cnn_test.ipynb`: Notebook for testing spectrograms with a simple CNN model.
- `keyword_spotting_main.ipynb`: Main Jupyter notebook containing the implementation of keyword spotting with the speech_commands_v0.02 dataset.
- `demo.ipynb`: Load the trained model, record an audio of 1 second with sample rate of 16k and predict the class which it belongs to.
