import os

import numpy as np

import keras
import tensorflow as tf

import utils
import networks.network as network

from keras import layers

# Create a compositor to extend Keras layers with gradient function
def GradExtender(base, call_fnc, grad_fnc):
    class LayerExtension(base):
        def __init__(self, *args, **kwargs):
            super(LayerExtension, self).__init__(*args, **kwargs)

    LayerExtension.call_grad = grad_fnc
    LayerExtension.call_base = LayerExtension.call
    LayerExtension.call = call_fnc 

    return LayerExtension

def LineGrad(self, outputs):
    grad_vals = np.zeros((outputs.shape[1], outputs.shape[1]))
    for i in range(outputs.shape[1]):
        grad_vals[i,i] = 1.0

    if self.use_bias:
        gradient = self.weights[0] @ tf.Variable(grad_vals, dtype="float32") + self.weights[1]
    else:
        gradient = self.weights[0] @ tf.Variable(grad_vals, dtype="float32")

    return gradient

def ReluGrad(self, outputs):
    grad_vals = np.zeros((outputs.shape[1], outputs.shape[1]))
    for i in range(outputs.shape[1]):
        if outputs[0,i] > 0:
            grad_vals[i,i] = 1.0

    if self.use_bias:
        gradient = self.weights[0] @ tf.Variable(grad_vals, dtype="float32") + self.weights[1]
    else:
        gradient = self.weights[0] @ tf.Variable(grad_vals, dtype="float32")

    return gradient

def CallWithGrads(self, inputs):
    outputs = self.call_base(inputs)
    self.gradient = self.call_grad(outputs)
    return outputs

# Create GradientLayers Protos
LineLayerProto = GradExtender(keras.layers.Dense, CallWithGrads, LineGrad)
ReLULayerProto = GradExtender(keras.layers.Dense, CallWithGrads, ReluGrad)

# Create a custom Model:
loss_tracker = keras.metrics.Mean(name="loss")
mse_metric = keras.metrics.MeanSquaredError(name="mse")

class GradModel(keras.Sequential):
    # Only mean square error of the data
    def diff_loss(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    # Norm
    def norm_loss(self, a, b):
        return tf.norm(a-b)/tf.norm(a)

    # Combiner mean square error of the data and the gradients using reduce_mean
    # in the sum
    def combined_loss(self, y_true, y_pred, g_true, g_pred):
        data_loss = self.diff_loss(y_true, y_pred)
        grad_loss = self.diff_loss(g_true, g_pred)

        return tf.math.reduce_mean(data_loss) + tf.math.reduce_mean(grad_loss)

    # Combined norm loss
    def combined_norm_loss(self, y_true, y_pred, g_true, g_pred):
        data_loss = self.norm_loss(y_true, y_pred)
        grad_loss = self.norm_loss(g_true, g_pred)

        return data_loss + grad_loss

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass
            y_pred = self(x, training=True)

            m_grad_pred = self.layers[0].gradient
            for i in range(1, len(self.layers)):
                m_grad_pred = m_grad_pred @ self.layers[i].gradient

            # # Compute our own loss
            loss = self.combined_loss(y, y_pred, m_grad, m_grad_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)

        # Update metrics
        mse_metric.update_state(m_grad, m_grad_pred)
        # mse_metric.update_state(x, y_pred)
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

    def create_encoder(self, decoded_dim, encoded_dim):

        encoder = LineLayerProto(encoded_dim,  activation="linear", name="LinearDense1", use_bias=False)

        return encoder

    def create_decoder(self, decoded_dim, encoded_dim):

        decoder = LineLayerProto(decoded_dim, activation="linear", name="LinearDenseF", use_bias=False)

        return decoder

    def create_autoencoder(self, encoder, decoder):

        model = GradModel(
            [
                  encoder
                , decoder
            ]
        )

        return model

    def define_network(self, input_data, custom_loss):
        data = np.transpose(input_data)

        ## Create our NN
        encoder = self.create_encoder(1024, 5)
        decoder = self.create_decoder(1024, 5)

        autoencoder = self.create_autoencoder(encoder, decoder)

        autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.1, amsgrad=False))

        return autoencoder

    def encode_snapshot(self, encoder, snapshot):

        input_snap = (snapshot.T - self.data_min) / (self.data_max - self.data_min)
        encoded_snap = encoder.predict(input_snap) 

        return encoded_snap.T

    def decode_snapshot(self, decoder, encoded_snapshot):

        input_snap = encoded_snapshot.T
        decoded_snap = decoder.predict(input_snap) * (self.data_max - self.data_min) + self.data_min

        return decoded_snap.T

    def predict_snapshot(self, network, snapshot):

        input_snap = (snapshot.T - self.data_min) / (self.data_max - self.data_min)
        predicted_snap = network.predict(input_snap) * (self.data_max - self.data_min) + self.data_min

        return predicted_snap.T

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None