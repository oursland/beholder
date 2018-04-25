from keras import applications
from keras import optimizers
from keras.callbacks import History, ModelCheckpoint, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import keras.backend as K

import numpy as np

import cv2

import csv

def readFaces(csvfile, image_dir, dim):
    data = []
    labels = []

    # open the file with 'ilename, score pairs
    with open(csvfile) as f:
        linereader = csv.reader(f, delimiter=' ')
        for line in linereader:
            score = float(line[1])
            image = cv2.imread(image_dir + '/' + line[0])

            # perform face detection
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
            faces = face_cascade.detectMultiScale(gray, minSize=(50, 50))

            # for each face in image
            for (x, y, w, h) in faces:
                # crop to only the face
                cropped = image[y:y+h, x:x+w]
                image = cv2.resize(cropped, dim)
                image = img_to_array(image)

                data.append(image)
                labels.append(score)

    # convert data to numpy
    data = np.array(data, dtype="float") / 255.0

    # rescale the scores from 1-5 to 1-10
    labels = np.array(labels, dtype="float")
    labels = labels - labels.min()
    labels = (labels / labels.max()) * 9.0 + 1.0

    return data, labels

# dimensions of our images.
width, height = 224, 224

epochs = 30
batch_size = 64

# read in the datasets
train_data, train_labels = readFaces('data/train.txt', 'data/images', (width, height))
validation_data, validation_labels = readFaces('data/test.txt', 'data/images', (width, height))

# create model and load weights
base_model = applications.MobileNet(weights='imagenet', include_top=False, input_shape=(width, height, 3))
model = Sequential()
model.add(base_model)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='tanh'))
model.add(Dense(1))
model.load_weights('train.h5')

# compile the model
model.compile(
    loss='mean_squared_error',
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=['mae'])

# create a data augmentation system
augmentation = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")
augmentation.fit(train_data)

model.summary()

checkpoint = ModelCheckpoint('fine-tune.{epoch:02d}-{val_loss:.2f}.hdf5')
tensorboard = TensorBoard(
    log_dir='./logs/fine-tune',
    histogram_freq=1,
    batch_size=batch_size)

# train entire network
history = model.fit_generator(
    augmentation.flow(train_data, train_labels, batch_size),
    validation_data=(validation_data, validation_labels),
    epochs=epochs,
    steps_per_epoch=len(train_data) // batch_size,
    callbacks=[checkpoint, tensorboard],
    verbose=2)

print('History: ' + str(history.history))

model.save_weights('fine-tune.h5')
