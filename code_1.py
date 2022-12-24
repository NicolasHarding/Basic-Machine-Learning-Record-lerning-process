
# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#install data set for training
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#do sth but i haven't known, hope it will be clear in future
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

#show picture in data set, wow!!!
#my_slice = train_images[1, 5 : , 5 : ]

#import matplotlib.pyplot as plt
#plt.imshow(my_slice, cmap=plt.cm.binary)
#plt.show()
    

network.compile(optimizer='rmsprop',
                 
                loss='categorical_crossentropy',
                 
                metrics=['accuracy'])

#preprocessing data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#encode the labels, will be explined for me in chapter 3
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#train data
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)

















