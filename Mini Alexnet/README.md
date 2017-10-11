# Mini Alexnet
This folder contains code and saved models for mini Alexnet trained on CIFAR10

## Contents
  * ###  Code
    * [*cifar10_alexnet.py*](./cifar10_alexnet.py): Code to train mini Alexnet on CIFAR10 without regularisation.
    * [*cifar10_alexnet_rand_crop.py*](./cifar10_alexnet_rand_crop.py): Code to train mini Alexnet on CIFAR10 without regularisation and with random crop.
    * [*cifar10_alexnet_rand_labels.py*](./cifar10_alexnet_rand_labels.py): Code to train mini Alexnet on CIFAR10 without regularisation and with random training labels.
    * [*cifar10_alexnet_wd.py*](./cifar10_alexnet_wd.py): Code to train mini Alexnet on CIFAR10 with weight decay.
    * [*cifar10_alexnet_wd_rand_crop.py*](./cifar10_alexnet_wd_rand_crop.py): Code to train mini Alexnet on CIFAR10 with weight decay and with random crop.

  * ### Saved Models
    * *cifar10_alexnet.h5*: A saved Keras Sequential model for mini Alexnet trained on CIFAR10.
    * *cifar10_alexnet_rand_crop.h5*: A saved Keras Sequential model for mini Alexnet trained on CIFAR10 with random crop.
    * *cifar10_alexnet_rand_labels.h5*: A saved Keras Sequential model for mini Alexnet trained on CIFAR10 with random labels.
    * *cifar10_alexnet_wd.h5*: A saved Keras Sequential model for mini Alexnet with L2 regularisation, trained on CIFAR10.
    * *cifar10_alexnet_wd_rand_crop.h5*: A saved Keras Sequential model for mini Alexnet with L2 regularisation, trained on CIFAR10 with random crop.
