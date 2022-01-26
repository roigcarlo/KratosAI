import os

import numpy as np

import keras
import tensorflow as tf

import utils
import networks.network as network

from keras import layers

# Create a custom Model:
loss_tracker = keras.metrics.Mean(name="loss")
mse_metric = keras.metrics.MeanSquaredError(name="mse")

class GradModel(keras.Model):
    # Mean square error of the data
    def diff_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    # Absolute error of the data
    def abs_loss(self, y_true, y_pred):
        return tf.abs(y_true - y_pred)

    # Norm
    def norm_loss(self, a, b):
        return tf.norm(a-b)/tf.norm(a)

    def set_m_grad(self, m_grad):
        self.m_grad = m_grad

    def set_g_weight(self, w):
        self.grad_weight = w

    # Combiner mean square error of the data and the gradients using reduce_mean
    # in the sum
    def combined_loss(self, y_true, y_pred, g_true, g_pred):
        data_loss = self.diff_loss(y_true, y_pred)
        grad_loss = self.diff_loss(g_true, g_pred)

        # return tf.math.reduce_mean(tf.math.maximum(data_loss, grad_loss))
        return (1 - self.grad_weight) * tf.math.reduce_mean(data_loss) + (self.grad_weight) * tf.math.reduce_mean(grad_loss)
        # return tf.maximum(tf.math.reduce_mean(data_loss), tf.math.reduce_mean(grad_loss)*8)

    # Combined norm loss
    def combined_norm_loss(self, y_true, y_pred, g_true, g_pred):
        data_loss = self.norm_loss(y_true, y_pred)
        grad_loss = self.norm_loss(g_true, g_pred)

        return data_loss + grad_loss

    def train_step(self, data):
        x, g = data

        # Automatic Gradient
        with tf.GradientTape(persistent=True) as tape:
            y, e = self(x, training=True)
            loss = self.diff_loss(x, y)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Compute our own metrics
        loss_tracker.update_state(loss)

        # Update metrics
        mse_metric.update_state(x, y)
            
        return {"loss": loss_tracker.result(), "mse": mse_metric.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, mse_metric]

class GradientShallow(network.Network):

    def __init__(self):
        '''
        Initis the network class

        model_name:     name of the directory in which the model will be saved.
        valid:          percentage of the input which is valid. (First steps are typically)
        '''
        super().__init__()

        self.model_name = "./saved_models/gradient_shallow"
        self.valid = 0.8
        self.encoder = None

    def define_network(self, input_data, custom_loss, encoded_size):
        data = np.transpose(input_data)
        decoded_size = data.shape[1]

        print(f'{encoded_size=} and {decoded_size=}')

        self.model_input = tf.keras.Input(shape=(decoded_size,))

        self.encoder = tf.keras.layers.Dense(decoded_size * 3,      activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))  (self.model_input)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                                                 (self.encoder)
        self.encoder = tf.keras.layers.Dense(10,                    activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))  (self.encoder)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                                                 (self.encoder)
        
        self.encoder = tf.keras.layers.Dense(5,                     activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))  (self.encoder)

        self.decoder = tf.keras.layers.Dense(decoded_size - 100,    activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(self.encoder)
        self.decoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                                                 (self.decoder)
        self.decoder = tf.keras.layers.Dense(decoded_size * 1,      activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))  (self.decoder)

        self.encoder_model = GradModel(self.model_input, self.encoder)
        self.autoenco = GradModel(self.model_input, [self.decoder, self.encoder])
        self.autoenco.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.01988, amsgrad=True), run_eagerly=False)
        self.autoenco.summary()
        
        return self.autoenco, self.autoenco

    ''' Calculates the derivative of a neural network model respect of the input '''
    def get_gradients(self, model, models_variables, input_data, output_data):
        in_tf_var = tf.Variable(input_data)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(in_tf_var)
            output = model(in_tf_var, training=False)
        auto_grad = tape.batch_jacobian(output, in_tf_var, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False) 
        print(output.shape, in_tf_var.shape, auto_grad.shape)

        # Compute gradients
        return auto_grad[0].numpy()

    def encode_snapshot(self, encoder, snapshot):

        input_snap = (snapshot.T - self.data_min) / (self.data_max - self.data_min)
        encoded_snap = encoder.predict(input_snap) 

        return encoded_snap.T

    def decode_snapshot(self, decoder, encoded_snapshot):

        input_snap = encoded_snapshot.T
        decoded_snap = decoder.predict(input_snap) * (self.data_max - self.data_min) + self.data_min

        return decoded_snap.T

    def predict_snapshot(self, network, snapshot):
        a, e = network.predict(snapshot.T)
        print(e)

        return a.T

    def train_network(self, model, input_data, grad_data, num_files, epochs=1):
        model.grads = grad_data
        model.fit(
            input_data.T, grad_data.T,
            epochs=epochs,
            batch_size=1,
        )

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None