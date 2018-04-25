from keras import applications
from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt

import cv2

import sys

# dimensions of our images.
width, height = 224, 224

def rate_faces(filename, model):
    # read image and convert to RGB
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # perform face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, minSize=(50, 50))

    # for each face in the image
    for (x, y, w, h) in faces:
        # crop to only the face
        cropped = image[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (width, height))
        resized = img_to_array(resized)
        data = np.array(resized, dtype="float").reshape(1, width, height, 3) / 255.0

        # perform scoring
        score = model.predict(data, verbose=1)

        # draw rectangle and score
        image = cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            4)
        image = cv2.putText(
            image,
            "{0:.2f}".format(score[0][0]),
            (x+4, y+h-6),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            6)
        image = cv2.putText(
            image,
            "{0:.2f}".format(score[0][0]),
            (x+4, y+h-6),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2)

    plt.imshow(image)
    plt.show()

def main():
    if len(sys.argv) < 2:
        printf("Usage: " + sys.argv[0] + " <filename>")
        sys.exit(0)

    # create model and load weights
    base_model = applications.MobileNet(weights='imagenet', include_top=False, input_shape=(width, height, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten(input_shape=base_model.output_shape[1:]))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(1))
    model.load_weights('fine-tune.h5')

    # process each file in the command line arguments
    for filename in sys.argv[1:]:
        rate_faces(filename, model)

if __name__ == "__main__":
    main()
