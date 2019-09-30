import numpy as np
import os
import pickle

from keras_dataset_gen import BreathDataGenerator
from simple_CNN import SimpleCNN


import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
# from resnet import ResnetBuilder
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
import librosa
from sklearn.metrics import classification_report, confusion_matrix



# Configuration
BATCH_SIZE = 32
LIST_LABELS = ['normal', 'deep', 'strong']
N_CLASSES = len(LIST_LABELS)
print(LIST_LABELS)
LR = 3
N_EPOCHS = 30
INPUT_SIZE = (40, 126, 1)

SOURCE_DEV =  'D:/Do An/Datasets/Breath_datasets_wav/Training/developement/output/'
SOURCE_TEST =  'D:/Do An/Datasets/Breath_datasets_wav/Training/validation/output/'
BEST_MODEL_PATH = "D:/Do An/breath-deep/model/resnet/weights-improvement-30-1.08.hdf5"

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))
config = tf.ConfigProto(device_count = {'GPU': 1})
sess = tf.Session(config=config)


# Get devset
train_generator = BreathDataGenerator(
        SOURCE_DEV,
        list_labels=LIST_LABELS,
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=False)
N_TRAIN_SAMPLES = len(train_generator.wavs)
print("Train samples: {}".format(N_TRAIN_SAMPLES))

# Get testset
validation_generator = BreathDataGenerator(
        SOURCE_TEST,
        list_labels=LIST_LABELS,
        batch_size=BATCH_SIZE,
        dim=INPUT_SIZE,
        shuffle=False)
N_VALID_SAMPLES = len(validation_generator.wavs)
print("Validation samples: {}".format(N_VALID_SAMPLES))



# Model training 

MODEL_NAME = 'CNN'

if (MODEL_NAME == 'CNN'):
        model = SimpleCNN.build(input_shape=INPUT_SIZE, classes=N_CLASSES)
if (MODEL_NAME == 'RESNET'):
       model = ResnetBuilder.build_resnet_18(input_shape=INPUT_SIZE, num_outputs=N_CLASSES)
if (MODEL_NAME == 'MobileNet'):
        model = MobileNet(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), include_top=True, classes=N_CLASSES, weights=None)
if (MODEL_NAME == 'MobileNetV2'):
        model = MobileNetV2(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1), include_top=True, classes=N_CLASSES, weights=None)


model.summary()
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])


mode = 'TRAIN'
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
if mode == 'TRAIN':
        model.fit_generator(
                train_generator,
                steps_per_epoch=N_TRAIN_SAMPLES // BATCH_SIZE,
                initial_epoch=0,
                epochs=N_EPOCHS,
                validation_data=validation_generator,
                validation_steps=N_VALID_SAMPLES // BATCH_SIZE,
                # callbacks=callbacks_list,
                max_queue_size=6,
                workers=3,
                use_multiprocessing=False
        )
        model.save_weights(MODEL_NAME + ".h5")
