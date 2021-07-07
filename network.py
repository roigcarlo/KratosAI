import os

import numpy as np

import keras
import tensorflow as tf

from keras import layers

valid = 0.8

class KratosNN:

    def __init__(self):
        self.model_name = "./saved_models/shallow_autoencoder"

    def create_encoder(self, input_dim, output_dim):
        encoder_input = keras.Input(shape=(input_dim,), dtype='float64', name="encoder_input")

        x = layers.Dense(output_dim * 8, activation='relu', dtype='float64', name="encoder_hidden_1")(encoder_input)
        x = layers.Dense(output_dim * 4, activation='relu', dtype='float64', name="encoder_hidden_2")(x)
        x = layers.Dense(output_dim * 2, activation='relu', dtype='float64', name="encoder_hidden_3")(x)

        encoded = layers.Dense(output_dim, activation='relu', dtype='float64', name="encoder_output")(x)
        encoder = keras.Model(encoder_input, encoded, name = "encoder")

        return encoder

    def create_decoder(self, input_dim, output_dim):
        decoder_input = keras.Input(shape=(input_dim,), name="decoder_input")

        x = layers.Dense(input_dim * 2, activation='relu', dtype='float64', name="decoder_hidden_1")(decoder_input)
        # x = layers.Dense(output_dim / 3, activation='relu', dtype='float64', name="decoder_hidden_2.5")(x)
        x = layers.Dense(input_dim * 4, activation='relu', dtype='float64', name="decoder_hidden_2")(x)
        x = layers.Dense(input_dim * 8, activation='relu', dtype='float64', name="decoder_hidden_3")(x)

        decoded = layers.Dense(output_dim,  activation='sigmoid', dtype='float64', name="decoder_output")(x)
        decoder = keras.Model(decoder_input, decoded, name="decoder")

        return decoder

    def create_autoencoder(self, encoder, decoder, input_dim, output_dim):
        autoencoder_input = keras.Input(shape=(input_dim,), name="autodecoder_input")

        encoded = encoder(autoencoder_input)
        decoded = decoder(encoded)

        autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

        return autoencoder

    def define_network(self, input_data, custom_loss):
        data = np.transpose(input_data)

        ## Create our NN
        decoding_dim = data.shape[1]
        encoding_dim = 3

        encoder = self.create_encoder(decoding_dim, encoding_dim)
        decoder = self.create_decoder(encoding_dim, decoding_dim)

        autoencoder = self.create_autoencoder(encoder, decoder, decoding_dim, encoding_dim)

        autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.0001, amsgrad=False))

        return autoencoder

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

        train_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) <  (train_cut * valid)]
        valid_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) >= (train_cut * valid)]

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

    def predict_vector(self, autoencoder, input_vector):

        tmp = input_vector.reshape(1,len(input_vector))
        tmp = (tmp - self.data_min) / (self.data_max - self.data_min)
        predicted_vector = autoencoder.predict(tmp)
        predicted_vector = predicted_vector * (self.data_max - self.data_min) + self.data_min
        tmp2 = predicted_vector.T

        return np.asarray(predicted_vector.T)

    def encode_snapshot(self, encoder, snapshot):

        input_snap = (snapshot.T - self.data_min) / (self.data_max - self.data_min)

        encoded_snap = encoder.predict(input_snap) 

        return encoded_snap.T

    def decode_snapshot(self, decoder, encoded_snapshot):

        input_snap = encoded_snapshot.T

        decoded_snap = decoder.predict(input_snap) * (self.data_max - self.data_min) + self.data_min

        return decoded_snap.T

    def predict_snapshot(self, autoencoder, snapshot):

        input_snap = (snapshot.T - self.data_min) / (self.data_max - self.data_min)

        predicted_snap = autoencoder.predict(input_snap) * (self.data_max - self.data_min) + self.data_min

        return predicted_snap.T

    def calculate_gradients(self, input_data, encoder, decoder, lossf, trainable_variables):
        encoder_in = tf.constant([input_data])
    
        with tf.GradientTape() as tape:
            encoder_out = encoder(encoder_in)
            decoder_out = decoder(encoder_out)
        
            loss = lossf(encoder_in, decoder_out)

        grad = tape.gradient(loss, trainable_variables)

        return grad

    def compute_full_gradient(self, autoencoder, all_gradients):
        gradient = None
        for var, g in zip(autoencoder.trainable_variables, all_gradients):
            if 'kernel' in var.name:
                if gradient is None:
                    gradient = g
                else:
                    gradient = gradient@g 
            print(f'{var.name}, shape: {g.shape}', gradient.shape)

        return gradient

    def generate_perturbation(self, encoded_snapshot):

        pert_sample = np.random.rand(encoded_snapshot.shape[0])
        pert_magnitude = np.linalg.norm(encoded_snapshot)

        return pert_sample * pert_magnitude

    def check_gradient(self, SEncoded, SDecoded):
        # Generate a perturbations for a solution in the compressed space (dq)
        norm_dq2 = []
        norm_qdq = []

        q = SEncoded[:,snapshot_index]
        per = self.generate_perturbation(q)

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