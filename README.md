# Deep Learning Generalization
My project to recreate the results in "Understanding Deep Learning Requires Rethinking Generalization" and do something similar on NLP.

## Contents
* ###  Code
  * *cifar10_alexnet.py*: Code to train mini Alexnet on CIFAR10 without regularisation.
  * *cifar10_alexnet_wd.py*: Code to train mini Alexnet on CIFAR10 with weight decay.
  * *cifar10_alexnet_rand_labels.py*: Code to train mini Alexnet on CIFAR10 without regularisation and with random training labels.
  * *cifar10_keras_mlp3.py*: Code to train MLP with 3 hidden layers on CIFAR10 without regularisation.
  * *cifar10_keras_mlp3_wd.py*: Code to train MLP with 3 hidden layers on CIFAR10 with weight decay.
  * *cifar10_keras_mlp3_rand.py*: Code to train MLP with 3 hidden layers on CIFAR10 without regularisation and with random training labels.
  * *cifar10_keras_mlp.py*: Code to train MLP with 1 hidden layer on CIFAR10 without regularisation.
  * *cifar10_keras_mlp_wd.py*: Code to train MLP with 1 hidden layer on CIFAR10 with weight decay.
  * *cifar10_keras_mlp_rand.py*: Code to train MLP with 1 hidden layer on CIFAR10 without regularisation and with random training labels.
  * *cifar10_tf_inception.py*: Code to train mini Inception on CIFAR10 without regularisation.

* ### Saved Models
  * *cifar10_alexnet.h5*: A saved Keras Sequential model for mini Alexnet trained on CIFAR10.
  * *cifar10_alexnet.h5*: A saved Keras Sequential model for mini Alexnet with L2 regularisation, trained on CIFAR10.
  * *cifar10_keras_mlp3.h5*: A saved Keras Sequential model for MLP with 3 hidden layers, trained on CIFAR10.
  * *cifar10_keras_mlp3_wd.h5*: A saved Keras Sequential model for MLP with 3 hidden layers and L2 regularisation, trained on CIFAR10.
  * *cifar10_keras_mlp3_rand.h5*: A saved Keras Sequential model for MLP with 3 hidden layers, trained on CIFAR10 with random labels.
  * *cifar10_keras_mlp.h5*: A saved Keras Sequential model for MLP with 1 hidden layer, trained on CIFAR10.
  * *cifar10_keras_mlp_wd.h5*: A saved Keras Sequential model for MLP with 1 hidden layer and L2 regularisation, trained on CIFAR10.
  * *cifar10_keras_mlp_rand.h5*: A saved Keras Sequential model for MLP with 1 hidden layer, trained on CIFAR10 with random labels.

* ### Folders
  * *results*: A folder containing screenshots of my current best results
