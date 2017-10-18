#!/bin/python2
import tensorflow as tf
import numpy as np
import pickle
import sys
import time
from operator import mul
from random import randint

def weight_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()(shape)
    #initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.contrib.layers.xavier_initializer()(shape)
    #initial = tf.truncated_normal(shape, mean=0.1)
    return tf.Variable(initial)

def max_pool_2d(x, kernel_size_int=2, stride_int=2):
    return tf.nn.max_pool(x, ksize=[1, kernel_size_int, kernel_size_int, 1],
                          strides = [1, stride_int, stride_int, 1],
                          padding='SAME')

def mean_pool_2d(x, kernel_size_int=2, stride_int=2):
    return tf.nn.pool(x, window_shape=[kernel_size_int, kernel_size_int],
                      pooling_type='AVG', padding='SAME',
                      strides = [stride_int, stride_int])

def conv_module(x, input_filters, filters, kernel_size_int, stride_int):
    W = weight_variable([kernel_size_int, kernel_size_int, input_filters,
                         filters])
    b = bias_variable([filters])
    conv_output = tf.nn.conv2d(x, W, strides=[1, stride_int, stride_int, 1],
                               padding='SAME') + b
    norm_output = tf.layers.batch_normalization(conv_output)
    return tf.nn.relu(norm_output)

def inception_module(x, input_filters, ch1_filters, ch3_filters):
    ch1_output = conv_module(x, input_filters, ch1_filters, 1, 1)
    ch3_output = conv_module(x, input_filters, ch3_filters, 3, 1)
    return tf.concat([ch1_output, ch3_output], axis=-1)

def downsample_module(x, input_filters, ch3_filters):
    ch3_output = conv_module(x, input_filters, ch3_filters, 3, 2)
    pool_output = max_pool_2d(x, 3)
    return tf.concat([ch3_output, pool_output], axis=-1)

def dense(x, input_shape, num_neurons):
    if len(input_shape) > 2:
        flat = tf.reshape(x, [-1, reduce(mul, input_shape[1:], 1)])
    else:
        flat = x
    W = weight_variable([reduce(mul, input_shape[1:], 1), num_neurons])
    b = bias_variable([num_neurons])
    return tf.matmul(flat, W) + b

x = tf.placeholder(tf.float32, [None, 28, 28, 3])
y_actual = tf.placeholder(tf.float32, [None, 10])

# Network Architecture
layer1 = conv_module(x, 3, 96, 3, 1)
layer2 = inception_module(layer1, 96, 32, 32)
layer3 = inception_module(layer2, 32 + 32, 32, 48)
layer4 = downsample_module(layer3, 32 + 48, 80)
layer5 = inception_module(layer4, 80 + 32 + 48, 112, 48)
layer6 = inception_module(layer5, 112 + 48, 96, 64)
layer7 = inception_module(layer6, 96 + 64, 80, 80)
layer8 = inception_module(layer7, 80 + 80, 48, 96)
layer9 = downsample_module(layer8, 48 + 96, 96)
layer10 = inception_module(layer9, 96 + 48 + 96, 176, 160)
layer11 = inception_module(layer10, 176 + 160, 176, 160)
layer12 = mean_pool_2d(layer11, 7, 1)
y_pred = dense(layer12, [None, 7, 7, 176 + 160], 10)
#y_pred = tf.nn.softmax(y_pred)

#loss = tf.reduce_mean(tf.square(y_actual - y_pred))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_actual,
                                                              logits=y_pred))
tf.summary.scalar('loss', loss)

# The SGD Optimizer with momentum
learning_rate = tf.placeholder(tf.float32, [])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.MomentumOptimizer(learning_rate,
                                            momentum=0.9).minimize(loss)
    #train_step = tf.train.AdamOptimizer().minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
tf.summary.scalar('acc', accuracy)

img = tf.placeholder(tf.float32, [28, 28, 3])
norm_image = tf.image.per_image_standardization(img)

sess = tf.InteractiveSession()
tensorboard_data = tf.summary.merge_all()
current_time = str(time.time())
train_writer = tf.summary.FileWriter('../Tensorboard/inception/train/' +\
                                     current_time, sess.graph)
test_writer = tf.summary.FileWriter('../Tensorboard/inception/test/' +\
                                    current_time, sess.graph)
tf.global_variables_initializer().run()

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

def cifar_next_batch_train(batch_size):
    for i in range(0, len(cifar10_train_labels), batch_size):
        yield cifar10_train_images[i:(i + batch_size)],\
              cifar10_train_labels[i:(i + batch_size)]

# Hyperparameters
learn_rate = 0.1
decay_rate = 0.95
num_epochs = 50
batch_size = 1
display_every = 1
early_stop_threshold = 0.0001
early_stop_patience = 5

for i in range(num_epochs):
    if i == 0:
        patience = 0
        prev_loss = 0
    else:
        prev_loss = total_train_loss
    total_train_loss = 0
    total_train_accuracy = 0
    initial_time = time.time()
    for j, (batch_x, batch_y) in enumerate(cifar_next_batch_train(batch_size)):
        train_loss, train_acc, _, data = sess.run([loss, accuracy, train_step,
                                                   tensorboard_data],
                                                   feed_dict={x:batch_x,
                                                   y_actual:batch_y,
                                                   learning_rate:learn_rate})
        total_train_loss += train_loss
        total_train_accuracy += train_acc
        train_writer.add_summary(data, (i + 1) * (j + 1))
        if j % display_every == 0:
            time_left = ((time.time() - initial_time) / (j + 1)) * ((len(
                        cifar10_train_labels) / batch_size) - (j + 1))
            sys.stdout.write("\rEpoch: %2d, Loss: %7.5f, Accuracy:%6.2f%%, "\
                             "ETA: %4ds" % (i + 1, total_train_loss / (j + 1),
                             total_train_accuracy / (j + 1), time_left))
            sys.stdout.flush()
    print "\rEpoch: %2d, Loss: %7.5f, Accuracy:%6.2f%%, Time Taken: %4ds" % (
          i + 1, total_train_loss / (j + 1), total_train_accuracy / (j + 1),
          time.time() - initial_time)
    if ((prev_loss - total_train_loss) / (j + 1)) < early_stop_threshold:
        patience += 1
        if patience >= early_stop_patience:
            break
    else:
        patience = 0
    learn_rate *= decay_rate

del cifar10_train_images, cifar10_train_labels
print "Loading test images..."
cifar10_test_images = []
cifar10_test_labels = []
test_file = open('../../cifar-10-batches-py/test_batch', 'r')
test_dict = pickle.load(test_file)
for image, label in zip(test_dict['data'], test_dict['labels']):
    image_red = np.reshape(image[:1024], (32, 32))[2:-2, 2:-2] / 255.0
    image_red = np.reshape(image_red, (28, 28, 1))
    image_green = np.reshape(image[1024:2048], (32, 32))[2:-2, 2:-2] / 255.0
    image_green = np.reshape(image_green, (28, 28, 1))
    image_blue = np.reshape(image[2048:3072], (32, 32))[2:-2, 2:-2] / 255.0
    image_blue = np.reshape(image_blue, (28, 28, 1))
    image = np.concatenate([image_red, image_green, image_blue], axis=-1)
    image = norm_image.eval(feed_dict={img:image})
    cifar10_test_images.append(image)
    label = np.identity(10)[label]
    cifar10_test_labels.append(label)
test_file.close()

def cifar_next_batch_test(batch_size):
    for i in range(0, len(cifar10_test_labels), batch_size):
        yield cifar10_test_images[i:(i + batch_size)],\
              cifar10_test_labels[i:(i + batch_size)]

total_test_accuracy = 0
initial_time = time.time()
for j, (batch_x, batch_y) in enumerate(cifar_next_batch_test(batch_size)):
    test_acc, data = sess.run([accuracy, tensorboard_data], feed_dict={
                              x:batch_x, y_actual:batch_y})
    total_test_accuracy += test_acc
    test_writer.add_summary(data, j + 1)
    if j % display_every == 0:
        time_left = ((time.time() - initial_time) / (j + 1)) * ((len(
                    cifar10_test_labels) / batch_size) - (j + 1))
        sys.stdout.write("\rTest Accuracy: %5.2f%%, ETA: %4ds" % (
                         total_test_accuracy / (j + 1), time_left))
        sys.stdout.flush()
print "\rTest Accuracy:%6.2f%%, Time Taken: %4ds" % (total_test_accuracy / (
      j + 1), time.time() - initial_time)

