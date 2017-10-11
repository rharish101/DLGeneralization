#!/bin/python2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD
from keras.initializers import Constant
import keras.backend as K
import pickle
import time
from shutil import copy
from copy import deepcopy
from random import randint

model = Sequential()
model.add(Flatten(input_shape=(28, 28, 3)))
model.add(Dense(512, kernel_initializer='glorot_normal',
                bias_initializer=Constant(0.1), activation='relu'))
model.add(Dense(10, kernel_initializer='glorot_normal',
                bias_initializer=Constant(0.1), activation='softmax'))

early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5)
now = str(time.time())
tb_callback = TensorBoard(log_dir='../Tensorboard/mlp1/' + now)

img = tf.placeholder(tf.float32, [28, 28, 3])
norm_image = tf.image.per_image_standardization(img)

opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

cifar10_train_images = []
cifar10_train_labels = []
print "Loading training images..."
for i in range(1, 6):
    train_file = open('../../cifar-10-batches-py/data_batch_' + str(i), 'r')
    train_dict = pickle.load(train_file)
    for image, label in zip(train_dict['data'], train_dict['labels']):
        image_red = np.reshape(image[:1024], (32, 32))[2:-2, 2:-2] / 255.0
        image_red = np.reshape(image_red, (28, 28, 1))
        image_green = np.reshape(image[1024:2048], (32, 32))[2:-2,
                                                             2:-2] / 255.0
        image_green = np.reshape(image_green, (28, 28, 1))
        image_blue = np.reshape(image[2048:3072], (32, 32))[2:-2, 2:-2] / 255.0
        image_blue = np.reshape(image_blue, (28, 28, 1))
        image = np.concatenate([image_red, image_green, image_blue], axis=-1)
        image = norm_image.eval(feed_dict={img:image})
        cifar10_train_images.append(image)
        label = np.identity(10)[randint(0, 9)]
        cifar10_train_labels.append(label)
    train_file.close()

epochs = 200
batch_size = 2

prev_loss = 1e4
patience = deepcopy(early_stop.patience)
for epoch in range(epochs):
    hist = model.fit(np.array(cifar10_train_images), np.array(
                     cifar10_train_labels), epochs=(epoch + 1),
                     batch_size=batch_size, initial_epoch=epoch,
                     callbacks=[tb_callback])
    K.set_value(opt.lr, 0.95 * K.get_value(opt.lr))
    if hist.history[early_stop.monitor][0] - prev_loss > early_stop.min_delta:
        patience -= 1
    else:
        patience = deepcopy(early_stop.patience)
    if patience <= 0:
        break
    else:
        prev_loss = hist.history[early_stop.monitor][0]

del cifar10_train_images, cifar10_train_labels
print "Loading test images..."
cifar10_test_images = []
cifar10_test_labels = []
test_file = open('../../cifar-10-batches-py/test_batch', 'r')
test_dict = pickle.load(test_file)
for image, label in zip(test_dict['data'], test_dict['labels']):
    image_red = np.reshape(image[:1024], (32, 32))[2:-2, 2:-2] / 255.0
    image_red = np.reshape(image_red, (28, 28, 1))
    image_green = np.reshape(image[1024:2048], (32, 32))[2:-2,
                                                            2:-2] / 255.0
    image_green = np.reshape(image_green, (28, 28, 1))
    image_blue = np.reshape(image[2048:3072], (32, 32))[2:-2, 2:-2] / 255.0
    image_blue = np.reshape(image_blue, (28, 28, 1))
    image_blue = np.reshape(image_blue, (28, 28, 1))
    image = np.concatenate([image_red, image_green, image_blue], axis=-1)
    image = norm_image.eval(feed_dict={img:image})
    cifar10_test_images.append(image)
    label = np.identity(10)[label]
    cifar10_test_labels.append(label)
test_file.close()

print(model.evaluate(np.array(cifar10_test_images),
                     np.array(cifar10_test_labels), batch_size=256))

response = raw_input("Do you want to save this model? (Y/n): ")
if response.lower() not in ['n', 'no', 'nah', 'nein', 'nahi', 'nope']:
    model.save('cifar10_mlp1_rand.h5')
    copy('./cifar10_keras_mlp_rand.py', '../Tensorboard/mlp1/' + now)
    print "Model saved"

