import os

import numpy as np

import keras
import tensorflow as tf

import utils
import networks.network as network

from keras import layers

class DebugNetwork(network.Network):

    def __init__(self):
        '''
        Initis the network class

        model_name:     name of the directory in which the model will be saved.
        valid:          percentage of the input which is valid. (First steps are typically)
        '''
        super().__init__()

        self.model_name = "./saved_models/shallow_autoencoder"
        self.valid = 0.8

    def create_encoder(self, encoder_input, output_size):
        x = layers.Dense(output_size, activation='linear', dtype='float64', name="encoder_hidden_1")(encoder_input)

        encoder = keras.Model(encoder_input, x, name = "encoder")

        return encoder

    def define_network(self):

        ## Create our NN
        input_dim = 2

        # For this example are the same, but maybe we want to compare with clusters of a reduced dim diferent of the encoding dim
        output_dim = 1

        encoder_input = keras.Input(shape=(1, 2), dtype='float64', name="input")

        encoder = self.create_encoder(encoder_input, output_dim)

        encoder.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=0.0001, amsgrad=False))

        return encoder

    def encode_snapshot(self, encoder, snapshot):

        input_snap = (snapshot.T - self.data_min) / (self.data_max - self.data_min)
        encoded_snap = encoder.predict(input_snap) 

        return encoded_snap.T

    def decode_snapshot(self, decoder, encoded_snapshot):

        input_snap = encoded_snapshot.T
        decoded_snap = decoder.predict(input_snap) * (self.data_max - self.data_min) + self.data_min

        return decoded_snap.T

    def predict_snapshot(self, network, input_data):
        inputn = []
        output = []

        for i in range(len(input_data)):
            snapshot = input_data[i]
            inputn.append((snapshot.T - self.data_min) / (self.data_max - self.data_min))
        
        prediction = network.predict(inputn)

        for i in range(len(prediction)):
            t_prediction = [p.T * (self.data_max - self.data_min) + self.data_min for p in prediction[i]]
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
            train_cut = len(data) / num_files
            train_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) <  (train_cut * self.valid)]
            valid_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) >= (train_cut * self.valid)]

            train_samples = np.array(train_pre)
            valid_samples = np.array(valid_pre)

            train_dataset.append(np.asarray(train_samples))
            valid_dataset.append(np.asarray(valid_samples))

        return train_dataset, valid_dataset

    def train_network(self, model, input_data, num_files):
        train_dataset, valid_dataset = self.prepare_data(input_data, num_files)

        # Shuffle the snapshots to prevent batches from the same clusters
        # np.random.shuffle(train_dataset)
        # np.random.shuffle(valid_dataset)

        print("TN", train_dataset[0].shape)
        print("TN", train_dataset[1].shape)

        # Train the model
        # model.fit(
        #     train_dataset, train_dataset,
        #     epochs=10,
        #     batch_size=1,
        #     shuffle=True,                                   # Probably not needed as we already shuffle.
        #     validation_data=(valid_dataset, valid_dataset),
        # )

    def loss_object(self, y_true, y_pred):

        input = y_true[0]
        predc = y_pred[0][0]
        error = y_pred[0][1]

        r_diff = input - predc
        e_diff = input - predc - error

        # y_diff = abs(y_true[0]-y_pred[0][0]+y_pred[0][1])    # Difference in the data
        # c_diff = abs(y_true[1]-y_pred[0][1])    # Difference in the clusters gradient

        r_diff = r_diff ** 2
        e_diff = e_diff ** 2
        # c_diff = c_diff ** 2

        return [r_diff, e_diff]
        # return 1 * tf.reduce_sum(y_diff) + 1 * tf.reduce_sum(c_diff)

    def loss_function(self, model, partial, inputs, targets, training):
        z_, w_ = model(inputs, training=True)
        e_ = partial(inputs[0], training=False)

        return self.loss_object(y_true=targets, y_pred=[z_, e_])

    def grad(self, model, partial, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss_function(model, partial, inputs, targets, training=True)

        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def custom_train(self, model, partial, input_data, num_files):
        train_dataset, valid_dataset = self.prepare_data(input_data, num_files)

        print("CT", train_dataset[0].shape)
        print("CT", train_dataset[1].shape)

        # Keep results for plotting
        train_loss_results = []
        train_accuracy_results = []

        num_epochs = 15

        # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        optimizer = tf.keras.optimizers.Adam(lr=0.00003, amsgrad=True)

        for epoch in range(num_epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.MeanSquaredError()

            # for x, y in zip(train_dataset[0], train_dataset[1]):
            # for i in range(np.shape(train_dataset[0])[0]):
            for i in np.random.permutation(np.shape(train_dataset[0])[0]):
                x = train_dataset[0][:i]
                y = train_dataset[1][:i]

                # print("====> Unziped entry", np.shape(x), np.shape(y))
                # Optimize the model
                loss_value, grads = self.grad(model, partial, [x, y], [x, y])
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Track progress
                epoch_loss_avg.update_state(loss_value)  # Add current batch loss
                # Compare predicted label to actual label
                # training=True is needed only if there are layers with different
                # behavior during training versus inference (e.g. Dropout).
                # epoch_accuracy.update_state([x, y], model([x, y], training=True))

            # End epoch
            train_loss_results.append(epoch_loss_avg.result())
            train_accuracy_results.append(epoch_accuracy.result())

            if epoch % 1 == 0:
                print("Epoch {:03d}: Loss: {:3e}, Accuracy: {:.3%}".format(
                    epoch,
                    epoch_loss_avg.result(),
                    -1
                    # epoch_accuracy.result()
                ))
