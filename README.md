# Deep Learning Generalization
My project to recreate the results in the paper "[Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530 "Understanding deep learning requires rethinking generalization")" and do a similar analysis on LSTMs in NLP.

## Contents
  * ###  Code
    * #### [Inception v3](./Inception%20v3)
      All code taken from [Tensorflow Models - Inception](https://github.com/tensorflow/models/tree/master/research/inception) and modified for my use.
    * #### [Mini Alexnet](./Mini%20Alexnet)
      * [*cifar10_alexnet.py*](./Mini%20Alexnet/cifar10_alexnet.py): Code to train mini Alexnet on CIFAR10 without regularization.
      * [*cifar10_alexnet_rand_crop.py*](./Mini%20Alexnet/cifar10_alexnet_rand_crop.py): Code to train mini Alexnet on CIFAR10 without regularization and with random crop.
      * [*cifar10_alexnet_rand_labels.py*](./Mini%20Alexnet/cifar10_alexnet_rand_labels.py): Code to train mini Alexnet on CIFAR10 without regularization and with random training labels.
      * [*cifar10_alexnet_wd.py*](./Mini%20Alexnet/cifar10_alexnet_wd.py): Code to train mini Alexnet on CIFAR10 with weight decay.
      * [*cifar10_alexnet_wd_rand_crop.py*](./Mini%20Alexnet/cifar10_alexnet_wd_rand_crop.py): Code to train mini Alexnet on CIFAR10 with weight decay and with random crop.
    * #### [Mini Inception](./Mini%20Inception)
      * [*cifar10_tf_inception.py*](./Mini%20Inception/cifar10_tf_inception.py): Code to train mini Inception on CIFAR10 without regularization.
      * [*cifar10_tf_inception_rand_crop.py*](./Mini%20Inception/cifar10_tf_inception_rand_crop.py): Code to train mini Inception on CIFAR10 without regularization and with random crop.
      * [*cifar10_tf_inception_wd.py*](./Mini%20Inception/cifar10_tf_inception_wd.py): Code to train mini Inception on CIFAR10 with weight decay.
      * [*cifar10_tf_inception_wd_rand_crop.py*](./Mini%20Inception/cifar10_tf_inception_wd_rand_crop.py): Code to train mini Inception on CIFAR10 with weight decay and with random crop.
    * #### [MLP1 - MLP with 1 hidden layer](./MLP1)
      * [*cifar10_keras_mlp.py*](./MLP1/cifar10_keras_mlp.py): Code to train MLP with 1 hidden layer on CIFAR10 without regularization.
      * [*cifar10_keras_mlp_rand.py*](./MLP1/cifar10_keras_mlp.py): Code to train MLP with 1 hidden layer on CIFAR10 without regularization and with random training labels.
      * [*cifar10_keras_mlp_wd.py*](./MLP1/cifar10_keras_mlp.py): Code to train MLP with 1 hidden layer on CIFAR10 with weight decay.
    * #### [MLP3 - MLP with 3 hidden layers](./MLP3)
      * [*cifar10_keras_mlp3.py*](./MLP3/cifar10_keras_mlp3.py): Code to train MLP with 3 hidden layers on CIFAR10 without regularization.
      * [*cifar10_keras_mlp3_rand.py*](./MLP3/cifar10_keras_mlp3_rand.py): Code to train MLP with 3 hidden layers on CIFAR10 without regularization and with random training labels.
      * [*cifar10_keras_mlp3_wd.py*](./MLP3/cifar10_keras_mlp3_wd.py): Code to train MLP with 3 hidden layers on CIFAR10 with weight decay.

  * ### Saved Models
    * #### [Mini Alexnet](./Mini%20Alexnet)
      * *cifar10_alexnet.h5*: A saved Keras Sequential model for mini Alexnet trained on CIFAR10.
      * *cifar10_alexnet_rand_crop.h5*: A saved Keras Sequential model for mini Alexnet trained on CIFAR10 with random crop.
      * *cifar10_alexnet_rand_labels.h5*: A saved Keras Sequential model for mini Alexnet trained on CIFAR10 with random labels.
      * *cifar10_alexnet_wd.h5*: A saved Keras Sequential model for mini Alexnet with L2 regularization, trained on CIFAR10.
      * *cifar10_alexnet_wd_rand_crop.h5*: A saved Keras Sequential model for mini Alexnet with L2 regularization, trained on CIFAR10 with random crop.
    * #### [MLP1 - MLP with 1 hidden layer](./MLP1)
      * *cifar10_keras_mlp.h5*: A saved Keras Sequential model for MLP with 1 hidden layer, trained on CIFAR10.
      * *cifar10_keras_mlp_rand.h5*: A saved Keras Sequential model for MLP with 1 hidden layer, trained on CIFAR10 with random labels.
      * *cifar10_keras_mlp_wd.h5*: A saved Keras Sequential model for MLP with 1 hidden layer and L2 regularization, trained on CIFAR10.
    * #### [MLP3 - MLP with 3 hidden layers](./MLP3)
      * *cifar10_keras_mlp3.h5*: A saved Keras Sequential model for MLP with 3 hidden layers, trained on CIFAR10.
      * *cifar10_keras_mlp3_rand.h5*: A saved Keras Sequential model for MLP with 3 hidden layers, trained on CIFAR10 with random labels.
      * *cifar10_keras_mlp3_wd.h5*: A saved Keras Sequential model for MLP with 3 hidden layers and L2 regularization, trained on CIFAR10.

  * ### [Results](./Results)
    * #### Mini Alexnet
      * [*Alexnet_Cifar10.png*](./Results/Alexnet_Cifar10.png): Mini Alexnet with no regularization.
      * [*Alexnet_Cifar10_rand_crop.png*](./Results/Alexnet_Cifar10_rand_crop.png): Mini Alexnet with no regularization and with random crop.
      * [*Alexnet_Cifar10_rand_labels.png*](./Results/Alexnet_Cifar10_rand_labels.png): Mini Alexnet with no regularization and with random labels.
      * [*Alexnet_Cifar10_wd.png*](./Results/Alexnet_Cifar10_wd.png): Mini Alexnet with weight decay.
      * [*Alexnet_Cifar10_wd_rand_crop.png*](./Results/Alexnet_Cifar10_wd_rand_crop.png): Mini Alexnet with weight decay and with random crop.
    * #### Mini Inception
      * [*Inception_Cifar10.png*](./Results/Inception_Cifar10.png): Mini Inception with no regularization.
      * [*Inception_Cifar10_rand_crop.png*](./Results/Inception_Cifar10_rand_crop.png): Mini Inception with no regularization and with random crop.
      * [*Inception_Cifar10_wd.png*](./Results/Inception_Cifar10_wd.png): Mini Inception with weight decay.
      * [*Inception_Cifar10_wd_rand_crop.png*](./Results/Inception_Cifar10_wd_rand_crop.png): Mini Inception with weight decay and with random crop.
    * #### MLP1 - MLP with 1 hidden layer
      * [*MLP1_Cifar10.png*](./Results/MLP1_Cifar10.png): MLP with 1 hidden layer with no regularization.
      * [*MLP1_Cifar10_rand_labels.png*](./Results/MLP1_Cifar10_rand_labels.png): MLP with 1 hidden layer with no regularization and with random labels.
      * [*MLP1_Cifar10_wd.png*](./Results/MLP1_Cifar10_wd.png): MLP with 1 hidden layer with weight decay.
    * #### MLP3 - MLP with 3 hidden layers
      * [*MLP3_Cifar10.png*](./Results/MLP3_Cifar10.png): MLP with 3 hidden layers with no regularization.
      * [*MLP3_Cifar10_rand_labels.png*](./Results/MLP3_Cifar10_rand_labels.png): MLP with 3 hidden layers with no regularization and with random labels.
      * [*MLP3_Cifar10_wd.png*](./Results/MLP3_Cifar10_wd.png): MLP with 3 hidden layers with weight decay.
