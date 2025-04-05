#=============================================
# BreathSense: Just a project to test the idea of having privacy gurantees within Raspiratory Sounds Schema. 
    # "Federated Learning for Respiratory Sound Analysis",
    # "Discovering Attention Methods with ICBHI Respiratory Sound Database",
    # Author: Ali Mahdavi,
    # Disclaimer: The codes will undergo massive changes during the time,
    # Dataset: https://bhichallenge.med.auth.gr/node/51,
    # Current Steps: 
        ## Create spectrograms from respiratory audio files
        ## Build and train a federated learning model with attention mechanisms (CNN-LSTM model architecture for respiratory sound classification)
        ## Adding LSH/ Ensuring Privacy Gurantees
    # If you want to give the code to AI for any enhancements, please note that in the hashing part it maight get nasty. 
#=============================================

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy
import datasketch
from typing import List, Dict, Tuple, Optional
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Constants
SAMPLE_RATE = 16000  # Hz
FRAME_LENGTH = 1024  # samples
HOP_LENGTH = 512  # samples
N_MFCC = 20  # Number of MFCC features
N_FFT = 1024  # FFT window size

# This part is AI-Generated, don't panic :)
class FeatureExtractor: 
    """Extract acoustic features from raw audio data."""
    
    def __init__(self, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                 n_fft=N_FFT, hop_length=HOP_LENGTH):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_features(self, audio_file: str) -> np.ndarray:
        """Extract MFCC features from an audio file."""
        try:
            y, sr = librosa.load(audio_file, sr=self.sr)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc,
                                         n_fft=self.n_fft, hop_length=self.hop_length)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                                 n_fft=self.n_fft,
                                                                 hop_length=self.hop_length)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                                   n_fft=self.n_fft,
                                                                   hop_length=self.hop_length)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                               n_fft=self.n_fft,
                                                               hop_length=self.hop_length)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y,
                                                                   frame_length=self.n_fft,
                                                                   hop_length=self.hop_length)
            
            # Normalize features
            mfccs_normalized = self._normalize(mfccs)
            sc_normalized = self._normalize(spectral_centroid)
            sb_normalized = self._normalize(spectral_bandwidth)
            sr_normalized = self._normalize(spectral_rolloff)
            zcr_normalized = self._normalize(zero_crossing_rate)
            
            features = np.vstack([mfccs_normalized,
                                  sc_normalized,
                                  sb_normalized,
                                  sr_normalized,
                                  zcr_normalized])
            
            return features
        
        except Exception as e:
            print(f"Error extracting features from {audio_file}: {e}")
            return None

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (features - mean) / std
    
# I added this but I'm not pretty sure if it's working  correctly, I mean some other methods shall be tested too. 

    def apply_privacy_filter(self, features: np.ndarray, epsilon: float = 0.5) -> np.ndarray:
        sensitivity = 1.0  # Assuming normalized features with max sensitivity of 1
        scale = sensitivity / epsilon
        noise = np.random.laplace(loc=0, scale=scale, size=features.shape)
        return features + noise

# This example usage parts all around the codes are for testing, please do not touch them!  
# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor()
    features = extractor.extract_features("path/to/audio/file.wav")
    print(features)
