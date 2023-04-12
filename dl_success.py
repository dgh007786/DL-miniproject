import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from tqdm import tqdm
from torchvision import models
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt

# Load the CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define the ResNet block
def residual_block(inputs, filters, strides):
    shortcut = inputs
    x = Conv2D(filters, kernel_size=3, strides=strides, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization()(x)
    if strides != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=strides)(inputs)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x

# Define the ResNet architecture
inputs = Input(shape=(32, 32, 3))
x = Conv2D(64, kernel_size=3, strides=1, padding="same")(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)
x = residual_block(x, filters=64, strides=1)
x = residual_block(x, filters=64, strides=1)
x = residual_block(x, filters=64, strides=1)
x = residual_block(x, filters=128, strides=2)
x = residual_block(x, filters=128, strides=1)
x = residual_block(x, filters=128, strides=1)
x = residual_block(x, filters=256, strides=2)
x = residual_block(x, filters=256, strides=1)
x = residual_block(x, filters=256, strides=1)
x = AveragePooling2D(pool_size=4)(x)
x = Flatten()(x)
outputs = Dense(10, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# Data augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

# Train the model
batch_size = 128
epochs = 2
steps_per_epoch = x_train.shape[0] // batch_size
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size), 
                    steps_per_epoch=steps_per_epoch, epochs=epochs, 
                    validation_data=(x_test, y_test))

#summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Acc')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)

model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
