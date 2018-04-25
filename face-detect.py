import numpy as np
import cv2

import matplotlib.pyplot as plt

import os
import sys

def detect_face_from_image_path(image_path):
    # read image and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # perform face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, minSize=(50, 50))

    print(image_path + ': faces detected: ', len(faces))
    if len(faces) < 1:
        plt.imshow(image)
        plt.show()
        return

    # for each face in the image
    for (x, y, w, h) in faces:
        # draw a rectangle around it
        image = cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            4)

    plt.imshow(image)
    plt.show()

def main():
    if len(sys.argv) < 2:
        printf("Usage: " + sys.argv[0] + " <filename>")
        sys.exit(0)

    for filename in sys.argv[1:]:
        detect_face_from_image_path(filename)

if __name__ == '__main__':
    main()
