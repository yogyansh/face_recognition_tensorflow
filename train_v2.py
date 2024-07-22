from architecture import *
import os
import cv2
import mtcnn
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model

###### paths and variables #########
face_data = 'Faces/'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data, face_names)
    if not os.path.isdir(person_dir):
        continue  # Skip non-directory entries

    encodes = []  # Reset encodes for each person

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        if not os.path.isfile(image_path):
            continue  # Skip non-file entries

        img_BGR = cv2.imread(image_path)
        if img_BGR is None:
            continue  # Skip if the image can't be read

        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        x = face_detector.detect_faces(img_RGB)
        if len(x) == 0:
            continue  # Skip if no face is detected

        x1, y1, width, height = x[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = img_RGB[y1:y2, x1:x2]

        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d)[0]
        encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_names] = encode

path = 'encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)
