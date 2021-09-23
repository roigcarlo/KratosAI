import os

import numpy as np

import keras
import tensorflow as tf

import utils
import networks.network as network

from keras import layers

class ClusterAutoencoder(network.Network):

    def __init__(self):
        '''
        Initis the network class

        model_name:     name of the directory in which the model will be saved.
        valid:          percentage of the input which is valid. (First steps are typically)
        '''
        super().__init__()

        self.model_name = "./saved_models/shallow_autoencoder"
        self.valid = 0.8

    def create_encoder(self, encoder_input, decoded_dim, encoded_dim):
        x = layers.Dense(decoded_dim, activation='relu', dtype='float64', name="encoder_hidden_1")(encoder_input)

        x = layers.Dense(decoded_dim/2, activation='relu', dtype='float64', name="encoder_hidden_2")(x)
        x = layers.Dense(decoded_dim/4, activation='relu', dtype='float64', name="encoder_hidden_3")(x)
        x = layers.Dense(decoded_dim/6, activation='relu', dtype='float64', name="encoder_hidden_4")(x)

        e = layers.Dense(decoded_dim, activation='relu', dtype='float64', name="e_error_hidden_1")(encoder_input)

        e = layers.Dense(decoded_dim/2, activation='relu', dtype='float64', name="e_error_hidden_2")(e)
        e = layers.Dense(decoded_dim/4, activation='relu', dtype='float64', name="e_error_hidden_3")(e)
        e = layers.Dense(decoded_dim/6, activation='relu', dtype='float64', name="e_error_hidden_4")(e)

        encoded = layers.Dense(encoded_dim, activation='relu', dtype='float64', name="encoder_output")(x)
        e_error = layers.Dense(encoded_dim, activation='relu', dtype='float64', name="e_error_output")(e)
        encoder = keras.Model(encoder_input, [encoded, e_error], name = "encoder")

        return encoder

    def create_decoder(self, encoder_input, decoder_input, d_error_input, decoded_dim, encoded_dim):
        x = layers.Dense(encoded_dim, activation='relu', dtype='float64', name="decoder_hidden_1")(decoder_input)

        x = layers.Dense(decoded_dim/6, activation='relu', dtype='float64', name="encoder_hidden_2")(x)
        x = layers.Dense(decoded_dim/4, activation='relu', dtype='float64', name="encoder_hidden_3")(x)
        x = layers.Dense(decoded_dim/2, activation='relu', dtype='float64', name="encoder_hidden_4")(x)

        e = layers.Dense(encoded_dim, activation='relu', dtype='float64', name="d_error_hidden_1")(d_error_input)

        e = layers.Dense(decoded_dim/6, activation='relu', dtype='float64', name="d_error_hidden_2")(e)
        e = layers.Dense(decoded_dim/4, activation='relu', dtype='float64', name="d_error_hidden_3")(e)
        e = layers.Dense(decoded_dim/2, activation='relu', dtype='float64', name="d_error_hidden_4")(e)

        decoded = layers.Dense(decoded_dim,  activation='sigmoid', dtype='float64', name="decoder_output")(x)
        d_error = layers.Dense(decoded_dim,  activation='sigmoid', dtype='float64', name="d_error_output")(e)

        d_error = layers.Add(dtype='float64', name="d_error")([encoder_input, -d_error])

        # d_error = layers.Add(dtype='float64', name="d_error")([encoder_input, -decoded])

        decoder = keras.Model([decoder_input, d_error_input], [decoded, d_error], name="decoder")

        return decoder

    def create_cluster_lane(self, decoded_dim, encoded_dim):
        cluster_input = keras.Input(shape=(encoded_dim,), name="cluster_input")
        cluster_output = layers.Lambda(lambda x: x, name="cluster_output")(cluster_input)

        cluster_lane = keras.Model(cluster_input, cluster_output, name="cluster_lane")

        return cluster_lane

    def create_autoencoder(self, encoder, decoder, decoded_dim, encoded_dim):
        autoencoder_input = keras.Input(shape=(decoded_dim,), dtype='float64', name="autodecoder_input")
        cluster_input = keras.Input(shape=(encoded_dim,), dtype='float64', name="cluster_input")

        encoded = encoder(autoencoder_input)
        decoded = decoder(encoded)

        clane = self.create_cluster_lane(encoded_dim, encoded_dim)

        claned = clane(cluster_input)

        autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

        return autoencoder

    def define_network(self, input_data, custom_loss):
        data = np.transpose(input_data)

        ## Create our NN
        decoded_dim = data.shape[1]

        # For this example are the same, but maybe we want to compare with clusters of a reduced dim diferent of the encoding dim
        encoded_dim = 5
        cluster_dim = 5

        encoder_input = keras.Input(shape=(decoded_dim,), dtype='float64', name="encoder_input")
        decoder_input = keras.Input(shape=(encoded_dim,), name="decoder_input")
        d_error_input = keras.Input(shape=(encoded_dim,), name="d_error_input")

        encoder = self.create_encoder(encoder_input, decoded_dim, encoded_dim)
        decoder = self.create_decoder(encoder_input, decoder_input, d_error_input, decoded_dim, encoded_dim)

        autoencoder = self.create_autoencoder(encoder, decoder, decoded_dim, encoded_dim)

        autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.0001, amsgrad=False))

        return autoencoder

    def encode_snapshot(self, encoder, snapshot):

        input_snap = (snapshot.T - self.data_min) / (self.data_max - self.data_min)
        encoded_snap = encoder.predict(input_snap) 

        return encoded_snap.T

    def decode_snapshot(self, decoder, encoded_snapshot):

        input_snap = encoded_snapshot.T
        decoded_snap = decoder.predict(input_snap) * (self.data_max - self.data_min) + self.data_min

        return decoded_snap.T

    def predict_snapshot(self, network, input_data):
        output = []
        
        input_data_n = (input_data.T - self.data_min) / (self.data_max - self.data_min)
        prediction = network.predict(input_data_n)

        for i in range(len(prediction)):
            t_prediction = prediction[i].T * (self.data_max - self.data_min) + self.data_min #  [p * (self.data_max - self.data_min) + self.data_min for p in prediction[i]]
            output.append(t_prediction)

        return output

    def calculate_gradients(self, input_data, encoder, decoder, lossf, trainable_variables):
        encoder_in = tf.constant([input_data])
    
        with tf.GradientTape() as tape:
            encoder_out = encoder(encoder_in)
            decoder_out = decoder(encoder_out)
        
            loss = lossf(encoder_in, decoder_out)

        grad = tape.gradient(loss, trainable_variables)

        return grad

    def compute_full_gradient(self, network, all_gradients):
        gradient = None
        for var, g in zip(network.trainable_variables, all_gradients):
            if 'kernel' in var.name:
                if gradient is None:
                    gradient = g
                else:
                    gradient = gradient@g 

        return gradient

    def check_gradient(self, SEncoded, SDecoded):
        """ Checks the correctness of a given gradient """

        # Generate a perturbations for a solution in the compressed space (dq)
        norm_dq2 = []
        norm_qdq = []

        q = SEncoded[:,snapshot_index]
        per = utils.generate_perturbation(q)

        epsilons = 16
        for i in range(0, epsilons):
            
            dq = per * ((1e-3)/(2**i))
            qdq = q + dq
            
            SDecoded_q = self.decode_snapshot(decoder, np.asarray([q]).T)
            SDecoded_qdq = self.decode_snapshot(decoder, np.asarray([qdq]).T)

            e1 = SDecoded_qdq - SDecoded_q
            e1 = e1.flatten()

            e2 = full_gradient.numpy().T@dq
            dq2 = dq ** 2

            norm_qdq.append(np.linalg.norm(e1-e2))
            norm_dq2.append(np.linalg.norm(dq)**2)

        bquad = [1/(2**(2*i)) for i in range(0,len(norm_qdq))]
        blin = [1/(2**i) for i in range(0,len(norm_qdq))]

        plt.plot(norm_qdq,norm_dq2)
        plt.plot(norm_qdq,blin)
        plt.plot(norm_qdq,bquad)
        plt.xscale('log')
        plt.yscale('log')
        
        # plt.show()

        # Print the results
        for i in range(1, epsilons):
            e1e2 = norm_qdq[i-1] / norm_qdq[i]
            dqd2 = norm_dq2[i-1] / norm_dq2[i]
            print("Gradient check for Epsilon-{}, q+dq: {:.2f}, dq2: {:.2f}, ???: {:.2f}".format(
                2**i, 
                e1e2, 
                dqd2,
                abs(dqd2 - (e1e2**2)) / dqd2
            ))

    def prepare_data(self, input_data, num_files):
        train_dataset = []
        valid_dataset = []

        for i in range(len(input_data)):
            data = np.transpose(input_data[i])

            print("Data shape:", data.shape)
            print("RAW - Min:", np.min(data), "Max:", np.max(data))

            data = (data - self.data_min) / (self.data_max - self.data_min)

            print("NOR - Min:", np.min(data), "Max:", np.max(data))

            # Select some of the snapshots to train and some others to validate
            # train_cut = len(data) / num_files
            # train_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) <  (train_cut * self.valid)]
            # valid_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) >= (train_cut * self.valid)]

            train_samples = np.array(data)
            valid_samples = np.array(data)

            # print("Train Samples B:", train_samples.shape)
            # train_samples = self.expand_dataset(train_samples, 5)
            # print("Train Samples A:", train_samples.shape)

            train_dataset.append(np.asarray(train_samples))
            valid_dataset.append(np.asarray(valid_samples))

        return train_dataset, valid_dataset

    def train_network(self, model, input_data, num_files):
        train_dataset, valid_dataset = self.prepare_data(input_data, num_files)

        # Train the model
        # model.fit(
        #     train_dataset, train_dataset,
        #     epochs=10,
        #     batch_size=1,
        #     shuffle=True,                                   # Probably not needed as we already shuffle.
        #     validation_data=(valid_dataset, valid_dataset),
        # )

    def loss_object(self, y_true, y_pred):
        with tf.GradientTape() as tape:
            grad = tape.gradient(y_pred, y_true)
            
        input = y_true
        predc = y_pred[0]
        error = y_pred[1]

        r_diff = abs(input - predc) + error
        e_diff = error ** 2

        r_diff = r_diff ** 2
        # e_diff = e_diff ** 2

        return [r_diff, e_diff]

    @tf.function
    def loss_function(self, inputs, targets):
        y_ = self.model(inputs, training=True)
        # e_ = partial(inputs[0], training=False)

        return self.loss_object(y_true=targets, y_pred=y_)

    @tf.function
    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss_function(inputs, targets)

        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def custom_train(self, model, partial, input_data, num_files):
        train_dataset, valid_dataset = self.prepare_data([input_data], num_files)

        print(input_data.shape)
        print(train_dataset[0].shape)
        print(train_dataset[0][0].shape)
        print(train_dataset[0][1].shape)

        # Keep results for plotting
        train_loss_results = []
        train_accuracy_results = []

        num_epochs = 10

        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        optimizer = tf.keras.optimizers.Adam(lr=0.001, amsgrad=True)

        self.model = model

        for epoch in range(num_epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.MeanSquaredError()

            # for x, y in zip(train_dataset[0], train_dataset[1]):
            # for i in range(np.shape(train_dataset[0])[0]):
            for i in np.random.permutation(np.shape(train_dataset[0])[0]):
                x = tf.convert_to_tensor([train_dataset[0][i]])

                # Optimize the model
                loss_value, grads = self.grad(x, x)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                epoch_accuracy.update_state(x, model(x, training=False))

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 1 == 0:
                print("Epoch {:03d}: Loss: {:3e}, Accuracy: {:.3%}".format(
                    epoch,
                    epoch_loss_avg.result(),
                    epoch_accuracy.result()
                ))
