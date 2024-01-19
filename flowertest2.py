import tensorflow as tf
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import pathlib as pl
import cv2 as cv

# Specify the path to your local dataset
dataset = r"D:\user\sockets_web\person"

# Create an ImageDataGenerator and specify the parameters
data_generator = ImageDataGenerator(rescale=1./255)

# Use the ImageDataGenerator to load and preprocess the images
data_dir = pl.Path(dataset)
image_count = len(list(data_dir.glob('*/*.jpg')))
image_data = data_generator.flow_from_directory(
    data_dir, target_size=(256, 256), batch_size=32, class_mode='categorical')

# training split
train_ds = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=123, image_size=(224, 224), batch_size=32)

# Testing or Validation split
test_ds = keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=123, image_size=(224, 224), batch_size=32)

class_names = train_ds.class_names

num_classes = len(class_names)
# model = Sequential([
#     layers.Rescaling(1./255, input_shape=(224, 224, 3)),
#     layers.Conv2D(16, 3, kernel_regularizer=keras.regularizers.l2(
#         0.0001), padding="same", activation="relu"),
#     layers.MaxPool2D(),
#     layers.Conv2D(32, 3, padding="same", activation="relu"),
#     layers.MaxPool2D(),
#     layers.Conv2D(64, 3, padding="same", activation="relu"),
#     layers.MaxPool2D(),
#     layers.Flatten(),
#     layers.Dense(128, kernel_regularizer=keras.regularizers.l2(
#         0.0001), activation="sigmoid"),
#     layers.Dropout(0.5),
#     layers.Dense(num_classes)
# ])

# model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(
#     from_logits=True), metrics=["accuracy"])
# # model.summary()

# epochs = 10
# history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)

# model.save("flowerclassifier.keras")

# # accuracy
# train_accuracy = history.history['accuracy']
# test_accuracy = history.history['val_accuracy']

# # loss
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# print(f"Test Accuracy: {test_accuracy[-1] * 100:.2f}%")
# print(f"Loss: {val_loss[-1]}")

# print(f"Train Accuracy: {train_accuracy[-1] * 100:.2f}%")
# print(f"Loss: {loss[-1]}")

# predict image
path = r"c:\Users\Rogers\Pictures\Saved Pictures\rogers\me.jpg"

# read the image
img = cv.imread(path)
# img = cv.fastNlMeansDenoisingColored(img, None, 20, 10, 7, 21)
img = cv.resize(img, (224, 224))
cv.imshow("pic", img)
cv.waitKey(0)

# load the modal
modal = keras.models.load_model("flowerclassifier.keras")

predictions = modal.predict(np.array([img]))
print(f"Predictions: {predictions}")

predicted_class_index = np.argmax(predictions)
top_4_indices = np.argsort(predictions[0])[::-1][:4]
top_4_confidences = predictions[0, top_4_indices]
confidence = predictions[0, predicted_class_index]

confidence_percentages = (np.exp(top_4_confidences) /
                          np.sum(np.exp(top_4_confidences))) * 100

label = class_names[predicted_class_index]

print(f"Prediction: {label} with {confidence}% confidence")

for i, (index, confidence) in enumerate(zip(top_4_indices, confidence_percentages), 1):
    label = class_names[index]
    print(f"Top {i} Prediction: {label} with {confidence:.2f}% confidence")
