import keras
from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# print version
print(keras.__version__)

# ----------------------------------------------------------------------------------------------------------------------
# load and check data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(len(train_images))
print(train_images.dtype)

digit = train_images[7]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

my_slice = train_images[10: 12, 7:-7, 7:-7]
print(my_slice.shape)
print(my_slice)

# ----------------------------------------------------------------------------------------------------------------------
# prepare data
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# ----------------------------------------------------------------------------------------------------------------------
# define model
network = models.Sequential()
network.add(layers.Dense(512,
                         activation='relu',
                         input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# ----------------------------------------------------------------------------------------------------------------------
# train model
network.fit(train_images,
            train_labels,
            epochs=5,
            batch_size=128)

# ----------------------------------------------------------------------------------------------------------------------
# evaluate model
test_loss, test_acc = network.evaluate(test_images, test_labels)
print()
print(test_acc)

