import json
import logging
import os
import time
import warnings

import librosa
import numpy as np
import pandas as pd
import pydub
import sklearn.preprocessing
from tqdm import tqdm

THEANO_FLAGS = ('device=cuda,'
                'floatX=float32,'
                'dnn.conv.algo_bwd_filter=deterministic,'
                'dnn.conv.algo_bwd_data=deterministic')

os.environ['THEANO_FLAGS'] = THEANO_FLAGS
os.environ['KERAS_BACKEND'] = 'theano'
maxSpecShape = 0

import keras
keras.backend.set_image_dim_ordering('th')
from keras.layers.convolutional import Conv2D as Conv
from keras.layers.convolutional import MaxPooling2D as Pool
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.regularizers import l2 as L2


from config import *


def to_one_hot(targets, class_count):
    """Encode target classes in a one-hot matrix.
    """
    one_hot_enc = np.zeros((len(targets), class_count))

    for r in range(len(targets)):
        one_hot_enc[r, targets[r]] = 1

    return one_hot_enc


def extract_segment(filename):
    """Get one random segment from a recording.
    """
    spec = np.load("/home/tad/Desktop/DA/DataAudio/feature/spectrogramImage/" + filename + ".spec.npy").astype('float32')

    length = np.shape(spec)[1]

    # print("np.shape(spec)[1] = ", np.shape(spec)[1])
    # print("np.shape(spec) = ", np.shape(spec))

    if(SEGMENT_LENGTH < length):
        offset = np.random.randint(0, length - SEGMENT_LENGTH)
        # print("offset = ", offset)
        # print( np.shape(spec)[1] - SEGMENT_LENGTH + 1)

        spec = spec[:, offset:offset + SEGMENT_LENGTH]

        # print("np.stack([spec]).shape = ", np.stack([spec]).shape)
    else:
        tmp = np.zeros((SEGMENT_LENGTH,))
        offset = np.random.randint(0, SEGMENT_LENGTH - length)
        # print("offset = ", offset)
        # print(np.shape(spec)[1] - SEGMENT_LENGTH + 1)
        tmp[offset:offset + length] = spec

        spec = tmp

        # print("np.stack([spec]).shape = ", np.stack([spec]).shape)

    return np.stack([spec])


def iterrows(dataframe):
    """Iterate over a random permutation of dataframe rows.
    """
    while True:
        for row in dataframe.iloc[np.random.permutation(len(dataframe))].itertuples():
            yield row


def iterbatches(batch_size, training_dataframe):
    """Generate training batches.
    """
    itrain = iterrows(training_dataframe)

    while True:
        X, y = [], []

        for i in range(batch_size):
            row = next(itrain)
            X.append(extract_segment(row.filename))
            y.append(le.transform([row.category])[0])

        X = np.stack(X)
        y = to_one_hot(np.array(y), len(labels))

        X -= AUDIO_MEAN
        X /= AUDIO_STD

        yield X, y


if __name__ == '__main__':
    np.random.seed(1)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # Load dataset
    meta = pd.read_csv("/home/tad/Desktop/DA/DataAudio/datasetcsv/result.csv")

    labels = pd.unique(meta.sort_values('category')['category'])
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)

    # Generate spectrograms
    logger.info('Generating spectrograms...')

    if not os.path.exists("/home/tad/Desktop/DA/DataAudio/feature/spectrogramImage/"):
        os.mkdir("/home/tad/Desktop/DA/DataAudio/feature/spectrogramImage/")


    for row in tqdm(meta.itertuples(), total=len(meta)):
        spec_file = "/home/tad/Desktop/DA/DataAudio/feature/spectrogramImage/" + row.filename + ".spec.npy"

        namefile = row.filename
        namefile = namefile.split(".")[0]
        audio_file = "/home/tad/Desktop/DA/DataAudio/data_wav_cut/" + namefile + ".wav"
        print(row.filename)

        if os.path.exists(spec_file):
            continue

        audio = pydub.AudioSegment.from_file(audio_file).set_frame_rate(SAMPLING_RATE).set_channels(1)
        audio = (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)

        spec = librosa.feature.melspectrogram(audio, SAMPLING_RATE, n_fft=FFT_SIZE,
                                              hop_length=CHUNK_SIZE, n_mels=MEL_BANDS)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Ignore log10 zero division
            spec = librosa.core.perceptual_weighting(spec, MEL_FREQS, amin=1e-5, ref_power=1e-5,
                                                     top_db=None)

        spec = np.clip(spec, 0, 100)

        np.save(spec_file, spec.astype('float32'), allow_pickle=False)

    # Define model
    logger.info('Constructing model...')

    input_shape = 1, MEL_BANDS, SEGMENT_LENGTH
    print(input_shape)

    model = keras.models.Sequential()

    model.add(Conv(80, (3, 3), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform',
                   input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Pool((3, 3), (3, 3)))

    model.add(Conv(160, (3, 3), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((3, 3), (3, 3)))

    model.add(Conv(240, (3, 3), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((3, 3), (3, 3)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(len(labels), kernel_regularizer=L2(0.001), kernel_initializer='he_uniform'))
    model.add(Activation('softmax'))

    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train model
    batch_size = 100
    EPOCH_MULTIPLIER = 10
    epochs = 1000 // EPOCH_MULTIPLIER
    epoch_size = len(meta) * EPOCH_MULTIPLIER
    bpe = epoch_size // batch_size

    logger.info('Training... (batch size of {} | {} batches per epoch)'.format(batch_size, bpe))

    model.fit_generator(generator=iterbatches(batch_size, meta),
                        steps_per_epoch=bpe,
                        epochs=epochs)

    with open('model.json', 'w') as file:
        file.write(model.to_json())

    model.save_weights('model.h5')

    with open('model_labels.json', 'w') as file:
        json.dump(le.classes_.tolist(), file)
