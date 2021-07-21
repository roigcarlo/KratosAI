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

    @abc.abstractmethod
    def calculate_gradients(self, input_data, encoder, decoder, lossf, trainable_variables):
        """ Compute the gradients of the input network list for the given input variables """

    @abc.abstractmethod
    def compute_full_gradient(self, network, all_gradients):
        """ Operates all gradients fir the given input variables """

    def check_gradient(self, SEncoded, SDecoded):
        """ Checks the correctness of a given gradient """
        print("Method not implemented.")

    def calculate_data_limits(self, input_data):
        self.data_min = np.min(input_data)
        self.data_max = np.max(input_data)

    def prepare_data(self, input_data):
        data = np.transpose(input_data)

        print("Data shape:", data.shape)
        print("RAW - Min:", np.min(data), "Max:", np.max(data))

        data = (data - self.data_min) / (self.data_max - self.data_min)

        print("NOR - Min:", np.min(data), "Max:", np.max(data))

        # Select some of the snapshots to train and some others to validate
        train_cut = len(data) / num_files
        train_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) <  (train_cut * self.valid)]
        valid_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) >= (train_cut * self.valid)]

        train_samples = np.array(train_pre)
        valid_samples = np.array(valid_pre)

        train_dataset = np.asarray(train_samples)
        valid_dataset = np.asarray(valid_samples)

        return train_dataset, valid_dataset

    def train_network(self, model, input_data, num_files):
        train_dataset, valid_dataset = self.prepare_data(input_data, num_files)

        # Shuffle the snapshots to prevent batches from the same clusters
        np.random.shuffle(train_dataset)
        np.random.shuffle(valid_dataset)

        # Train the model
        model.fit(
            train_dataset, train_dataset,
            epochs=50,
            batch_size=1,
            shuffle=True,                                   # Probably not needed as we already shuffle.
            validation_data=(valid_dataset, valid_dataset),
        )

    def train_with_gradient(self, network, loss, input_data, num_files):
        train_dataset, valid_dataset = self.prepare_data(input_data, num_files)

        # Shuffle the snapshots to prevent batches from the same clusters
        np.random.shuffle(train_dataset)
        np.random.shuffle(valid_dataset)

        # Train the model
        epochs = 2
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

                # Open a GradientTape to record the operations run
                # during the forward pass, which enables auto-differentiation.
                with tf.GradientTape() as tape:

                    # Run the forward pass of the layer.
                    # The operations that the layer applies
                    # to its inputs are going to be recorded
                    # on the GradientTape.
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch

                    # Compute the loss value for this minibatch.
                    loss_value = loss_fn(y_batch_train, logits)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Log every 200 batches.
                if step % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))

    def predict_vector(self, network, input_vector):

        tmp = input_vector.reshape(1,len(input_vector))
        tmp = (tmp - self.data_min) / (self.data_max - self.data_min)
        predicted_vector = network.predict(tmp)
        predicted_vector = predicted_vector * (self.data_max - self.data_min) + self.data_min
        tmp2 = predicted_vector.T

        return np.asarray(predicted_vector.T)