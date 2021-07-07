import os
import abc

import numpy as np

import keras
import tensorflow as tf

from keras import layers

class Network(abc.ABC):

    def __init__(self):
        """
        Initis the network class

        model_name:     name of the directory in which the model will be saved.
        valid:          percentage of the input which is valid. (First steps are typically)
        """

        self.model_name = "./saved_models/emtpy_network"
        self.valid = 1.0

    @abc.abstractmethod
    def define_network(self, input_data, custom_loss):
        """ Define the network configuration """

    def prepare_data(self, input_data):
        self.data_min = np.min(input_data)
        self.data_max = np.max(input_data)

    def train_network(self, model, input_data, num_files):
        data = np.transpose(input_data)

        print("Data shape:", data.shape)

        self.data_min = np.min(data)
        self.data_max = np.max(data)

        print("RAW - Min:", np.min(data), "Max:", np.max(data))
        
        data = (data - self.data_min) / (self.data_max - self.data_min)

        print("NOR - Min:", np.min(data), "Max:", np.max(data))

        unshuffled_data = data.copy()

        ### Training experiment ###
        train_cut = len(data) / num_files

        train_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) <  (train_cut * self.valid)]
        valid_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) >= (train_cut * self.valid)]

        train_samples = np.array(train_pre)
        valid_samples = np.array(valid_pre)

        train_dataset = np.asarray(train_samples)
        valid_dataset = np.asarray(valid_samples)

        np.random.shuffle(train_dataset)
        np.random.shuffle(valid_dataset)

        model.fit(
            train_dataset, train_dataset,
            epochs=50,
            batch_size=1, # int(370*0.7),
            shuffle=True,
            validation_data=(valid_dataset, valid_dataset),
        )

    def predict_vector(self, network, input_vector):

        tmp = input_vector.reshape(1,len(input_vector))
        tmp = (tmp - self.data_min) / (self.data_max - self.data_min)
        predicted_vector = network.predict(tmp)
        predicted_vector = predicted_vector * (self.data_max - self.data_min) + self.data_min
        tmp2 = predicted_vector.T

        return np.asarray(predicted_vector.T)

    @abc.abstractmethod
    def calculate_gradients(self, input_data, encoder, decoder, lossf, trainable_variables):
        """ Compute the gradients of the input network list for the given input variables """

    @abc.abstractmethod
    def compute_full_gradient(self, network, all_gradients):
        """ Operates all gradients fir the given input variables """

    def generate_perturbation(self, encoded_snapshot):

        pert_sample = np.random.rand(encoded_snapshot.shape[0])
        pert_magnitude = np.linalg.norm(encoded_snapshot)

        return pert_sample * pert_magnitude

    def check_gradient(self, SEncoded, SDecoded):
        """ Checks the correctness of a given gradient """
        print("Method not implemented.")