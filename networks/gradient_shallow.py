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

    gradient = self.weights[0] @ grad_vals

    return gradient

def ReluGrad(self, outputs):
    grad_vals = np.zeros((outputs.shape[1], outputs.shape[1]))
    for i in range(outputs.shape[1]):
        if outputs[0,i] > 0:
            grad_vals[i,i] = 1.0

    gradient = self.weights[0] @ grad_vals

    return gradient

def CallWithGrads(self, inputs):
    outputs = self.call_base(inputs)
    self.gradient = self.call_grad(outputs)
    return outputs

# Create GradientLayers Protos
LineLayerProto = GradExtender(keras.layers.Dense, CallWithGrads, LineGrad)
# LeakyReLULayerProto = GradExtender(keras.layers.LeakyRelu, CallWithGrads, ReluGrad)

# Create a custom Model:
loss_tracker = keras.metrics.Mean(name="loss")
mse_metric = keras.metrics.MeanSquaredError(name="mse")

class GradModel(keras.Sequential):
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

    # Combiner mean square error of the data and the gradients using reduce_mean
    # in the sum
    def combined_loss(self, y_true, y_pred, g_true, g_pred):
        data_loss = self.diff_loss(y_true, y_pred)
        grad_loss = self.diff_loss(g_true, g_pred)

        grad_weight = 1

        # return tf.math.reduce_mean(tf.math.maximum(data_loss, grad_loss))
        return (1 - grad_weight) * tf.math.reduce_mean(data_loss) + (grad_weight) * tf.math.reduce_mean(grad_loss)

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

            self.m_grad_pred = m_grad_pred

            # print(f'{m_grad_pred=}')
            # print(f'{(self.m_grad-m_grad_pred)**2=}')

            # Compute our own loss
            loss = self.combined_loss(y, y_pred, self.m_grad, m_grad_pred)
            # loss = self.diff_loss(self.m_grad, m_grad_pred)
            # loss = self.diff_loss(x, y_pred)

        # print("Loss:", loss)
        # tf.print("x", x)
        # tf.print("yp", y_pred)
        # exit(0)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)

        # Update metrics
        # mse_metric.update_state(m_grad, m_grad_pred)
        mse_metric.update_state(x, y_pred)
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

        encoder = LineLayerProto(encoded_dim,  activation="relu", name="LinearDense1", use_bias=True)

        return encoder

    def create_decoder(self, decoded_dim, encoded_dim):

        decoder = LineLayerProto(decoded_dim, activation="sigmoid", name="LinearDenseF", use_bias=True)

        return decoder

    def create_autoencoder(self, encoder, decoder):

        model = GradModel(
            [
                  encoder
                , decoder
            ]
        )

        return model

    def define_network(self, input_data, custom_loss, encoded_size):
        data = np.transpose(input_data)
        
        decoded_size = data.shape[1]
        # encoded_size = 54

        ## Create our NN
        encoder = self.create_encoder(decoded_size, encoded_size)
        decoder = self.create_decoder(decoded_size, encoded_size)

        autoencoder = self.create_autoencoder(encoder, decoder)

        autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.0025*1, amsgrad=True), run_eagerly=False)

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

        return (network.predict(snapshot.T)).T

    def train_network(self, model, input_data, grad_data, num_files, epochs=1):
        # Train the model
        model.grads = grad_data
        model.fit(
            input_data.T, input_data.T,
            epochs=epochs,
            batch_size=1,
            # shuffle=True,
            # validation_data=(valid_dataset, valid_dataset),
        )

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None