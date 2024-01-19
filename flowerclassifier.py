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

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(25):
#         ax = plt.subplot(5, 5, i+1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()
num_classes = len(class_names)
model = Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes)
])

model.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=["accuracy"])
# model.summary()

epochs = 10
history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)

model.save("flowerclassifier.keras")

# accuracy
train_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

# loss
loss = history.history['loss']
val_loss = history.history['val_loss']

print(f"Test Accuracy: {test_accuracy[-1] * 100:.2f}%")
print(f"Loss: {val_loss[-1]}")

print(f"Train Accuracy: {train_accuracy[-1] * 100:.2f}%")
print(f"Loss: {loss[-1]}")

# predict image
path = r"c:\Users\Rogers\Pictures\Saved Pictures\cathy\IMG-20231224-WA0019.jpg"

# read the image
img = cv.imread(path)
# convert the image to RGB
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (224, 224))

# load the modal
modal = keras.models.load_model("flowerclassifier.keras")

predictions = modal.predict(np.array([img]))

predicted_class_index = np.argmax(predictions)
confidence = predictions[0, predicted_class_index]

label = class_names[predicted_class_index]

print(f"Prediction: {label} with {confidence}% confidence")
