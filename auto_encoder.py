import numpy as np
import sklearn.prepocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorial.mnist import input_data

def xavier_init(fan_in, fan_out, constant = 1):
    pass

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,):
        pass

    def _initializer_weights(self):
        pass

    def partial_fit(self, x):
        pass

    def calc_total_cost(self, X):
        pass

    def transform(self, X):
        pass

    def generate(self, hidden = None):
        pass

    def reconstruct(self, X):
        pass

    def getWeights(self):
        pass

    def getBiases(self):
        pass

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

def standard_scale(X_train, X_test):
    pass

def get_random_block_from_data(data, batch_size):
    pass

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

autoencoder = 

for epoch in range(training_epochs):
    pass

