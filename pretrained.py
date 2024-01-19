import os
import cv2 as cv
import numpy as np
from testpy import extract_faces_dnn

people = []
DIR = r'D:\user\sockets_web\person'
for i in os.listdir(r'D:\user\sockets_web\person'):
    people.append(i)
features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)

            faces_rect = extract_faces_dnn(img_array)

            for face in faces_rect:
                gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
                features.append(gray)
                labels.append(label)


create_train()
print("------------------Training done ----------------------")
features = np.array(features, dtype="object")
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)
face_recognizer.save("face_trained.yml")
np.save('feautures.npy', features)
np.save("labels.npy", labels)


face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread(
    r"c:\Users\Rogers\Downloads\ben.jpg")

face_rect = extract_faces_dnn(img)

predictions = []

for face in face_rect:
    face_roi = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    label, confidence = face_recognizer.predict(face_roi)
    print(f"Label = {people[label]} with confidence of {confidence}")
    print(label)
    predictions.append({
        'label': people[label],
        'confidence': confidence
    })

predictions.sort(key=lambda x: x['confidence'], reverse=True)

# Retrieve the top 5 predictions
top_5_predictions = predictions[:5]

# Print the top 5 predictions
for i, prediction in enumerate(top_5_predictions):
    print(
        f"Rank {i+1}: Label = {prediction['label']} with confidence of {prediction['confidence']}")
