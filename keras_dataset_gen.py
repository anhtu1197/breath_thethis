import numpy as np
import keras
from scipy.io import wavfile
import librosa
import os
# from resnet import ResnetBuilder
BATCH_SIZE = 32
DIM = (40,126,1)
RESAMPLE = 8000
class BreathDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, directory, 
                    list_labels=['normal', 'deep', 'strong'], 
                    batch_size=128,
                    dim=None,
                    classes=None, 
                    shuffle=True):
        'Initialization'
        print("In Init function")

        self.directory = directory
        self.list_labels = list_labels
        self.dim = dim
        self.__flow_from_directory(self.directory)
        self.batch_size = batch_size
        self.classes = len(self.list_labels)
        self.shuffle = shuffle
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.wavs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        rawX = [self.wavs[k] for k in indexes]
        rawY = [self.labels[k] for k in indexes]

        # Generate data
        X, Y = self.__feature_extraction(rawX, rawY)
        # print("Done getting data")
        return X, Y

    def __flow_from_directory(self, directory):       
        #List of wav path
        self.wavs = []
        
        # List of labels of the file 
        self.labels = []
        filenames = os.listdir(directory)
        labels = []
        for filename in filenames:
            self.wavs.append(os.path.join(directory, filename))
            wav_label = (filename.split(".")[0]).split("-")[-1]
            self.labels.append(self.list_labels.index(wav_label))


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.wavs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __feature_extraction(self, list_wav, list_label):
        'Generates data containing batch_size samples'
        # Initialization
        X = []
        Y = []
        for i in range (BATCH_SIZE):
            data, rate= librosa.load(list_wav[i], mono=False)
            # data = librosa.to_mono(data)
            data = np.array(data, dtype=np.float32)
            data = librosa.resample(data, rate, RESAMPLE)
            data *= 1./32768
            if(len(data.shape) == 1):
                # Extract the feature 
                feature = librosa.feature.mfcc(y=data, sr=rate)
                feature = np.resize(feature, DIM)
                X.append(feature)
                Y.append(list_label[i])

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=int)
        return X, Y


