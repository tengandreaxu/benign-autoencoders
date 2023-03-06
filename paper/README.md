# Description

We collect here all scripts to reproduce the results
presented in our paper.

# Experiments

## MNIST

For MNIST we use the KerasMNISTAE.py Autoencoder. The Autoencoder was designed in the keras blog:  https://blog.keras.io/building-autoencoders-in-keras.html. We add a BatchNormalization layer at the end of each layer to smooth the loss landscape and escape the vanishing gradient descent problem. Finally, we use the BinaryCrossEntropy loss for the reconstruction error penalty and the canonical SparseCategoricalEntropy for the classification error.

## FMNIST

For FMNIST we use the KaggleMNISTAE.py. The Autoencoder was designed in the kaggle blog: https://www.kaggle.com/code/milan400/cifar10-autoencoder. We add BatchNormalization layer at the end of each layer to smooth the loss landscape and escape the vanishing gradint descent problem. Finally, we use the BinaryCrossEntropy loss for the reconstruction error penalty and the canonical SparseCategoricalEntropy for the classification error.