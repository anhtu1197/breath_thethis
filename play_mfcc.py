import numpy as np
import keras
from scipy.io import wavfile
import librosa
import os


rate, data = wavfile.read('D:/Do An/Datasets/Breath_datasets_wav/output/01_male_23_BQuyen0-strong.wav') #bug in here
data = np.array(data, dtype=np.float32)
data *= 1./32768

feature = librosa.feature.mfcc(y=data, sr=rate, 
                                n_mfcc=40, fmin=0, fmax=8000,
                                n_fft=int(16*64), hop_length=int(16*32), power=2.0)

feature = np.resize(feature, (40,126,1))

