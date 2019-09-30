
import numpy as np
import keras
from scipy.io import wavfile
import librosa
import os
from tqdm import tqdm
BATCH_SIZE = 32
DIM = (40,126,1)

DATA_SOURCE = 'D:/Do An/Datasets/Breath_datasets_wav/output'

def get_metadata_info(source_dataset):
    filenames = os.listdir(source_dataset)
    labels = []
    for filename in tqdm(filenames):
            wav_label = (filename.split(".")[0]).split("-")[-1]
            labels.append(wav_label)
    return filenames, labels


def mfcc_extraction(source_dataset, list_wav, list_label):
    # Initialization
    X = []
    Y = []
    for i in range (BATCH_SIZE):
        data, rate= librosa.load(list_wav[i], mono=False)
        # data = librosa.to_mono(data)
        data = np.array(data, dtype=np.float32)
        data *= 1./32768
        if(len(data.shape) == 1)
            # Extract the feature 
            feature = librosa.feature.mfcc(y=data, sr=rate)
            feature = np.resize(feature, DIM)
            X.append(feature)
            Y.append(wav_label[i])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=int)
    return X, Y





