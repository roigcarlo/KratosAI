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

    # LayerExtension.call_grad = grad_fnc
    # LayerExtension.call_base = LayerExtension.call
    # LayerExtension.call = call_fnc 

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
        return tf.math.reduce_sum(1 + abs(y_true - y_pred))

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

        # # Automatic Gradient
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape2:
            with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                tape.watch(x)
                y_pred = self(x, training=True)

                # self.m_grad_pred = self.layers[0].gradient
                # for i in range(1, len(self.layers)):
                #     self.m_grad_pred = self.m_grad_pred @ self.layers[i].gradient

                # with tape.stop_recording():
            # auto_grad = tape.batch_jacobian(y_pred, x, unconnected_gradients=tf.UnconnectedGradients.ZERO, experimental_use_pfor=False)

            # eval_y = x @ auto_grad
            # loss = self.combined_loss(x, eval_y, self.m_grad[0], auto_grad)
            loss = self.diff_loss(x, y_pred)

        # print(auto_grad[0].shape)
        # print(self.m_grad_pred.shape)
        # gnorm = np.linalg.norm(auto_grad[0]-self.m_grad_pred)/np.linalg.norm(self.m_grad_pred)

        # m_grad_pred = self.layers[0].gradient
        # for i in range(1, len(self.layers)):
        #     m_grad_pred = m_grad_pred @ self.layers[i].gradient

        # print(auto_grad)
        
        #tryvar = auto_grad - self.m_grad[0]
        # loss = tryvar

        # with tf.GradientTape(persistent=True) as tape:
        # print(tryvar-self.m_grad[0])
            # loss = self.diff_loss(x, y_pred)
            # loss = self.combined_loss(x, y_pred, self.m_grad[int(g)], auto_grad)

        # Manual Gradient (probalby bad)
        # with tf.GradientTape(persistent=True) as tape:
        #     # Forward pass
        #     y_pred = self(x, training=True)

        #     m_grad_pred = self.layers[0].gradient
        #     for i in range(1, len(self.layers)):
        #         m_grad_pred = m_grad_pred @ self.layers[i].gradient

        #     self.m_grad_pred = m_grad_pred

        #     # print(f'{m_grad_pred=}')
        #     # print(f'{(self.m_grad-m_grad_pred)**2=}')

        #     # Compute our own loss
        #     # print(f"Using cluster {int(g)=}")
        #     loss = self.combined_loss(x, y_pred, self.m_grad[int(g)], m_grad_pred)
        #     # loss = self.diff_loss(self.m_grad, m_grad_pred)
        #     # loss = self.diff_loss(x, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape2.gradient(loss, trainable_vars)

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

class GradModel2(keras.Model):
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

    def define_network(self, input_data, custom_loss, encoded_size):
        data = np.transpose(input_data)
        
        decoded_size = data.shape[1]
        # encoded_size = 54

        print(f'{encoded_size=} and {decoded_size=}')

        # tfcns = tf.keras.constraints.MinMaxNorm(
        #     min_value=5.0, max_value=10.0, rate=1.0
        # )

        tfcns = tf.keras.constraints.NonNeg()

        # tfinit = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

        # # # Create our NN
        # autoencoder = GradModel(
        #     [
        #         #   LineLayerProto(           10, activation="relu", name="Encode", use_bias=True, kernel_constraint=tfcns, bias_constraint=tfcns, kernel_initializer=tfinit)
        #           LineLayerProto(          10, activation="relu",    name="Encode", use_bias=True)
        #         , LineLayerProto(decoded_size, activation="sigmoid", name="Sigmoid", use_bias=True)
        #     ]
        # )

        # autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.01, amsgrad=True), run_eagerly=False)

        # return autoencoder, autoencoder

        self.model_input = tf.keras.Input(shape=(decoded_size,))

        self.encoder = tf.keras.layers.Dense(decoded_size * 4,  activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))  (self.model_input)
        self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                                                 (self.encoder)
        # self.encoder = tf.keras.layers.Dense(20 ,               activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))  (self.encoder)
        # self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                                                 (self.encoder)
        self.encoder = tf.keras.layers.Dense(5,                 activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))  (self.encoder)
        # self.encoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                                                 (self.encoder)
        
        self.decoder = tf.keras.layers.Dense(decoded_size * 1,  activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(self.encoder)
        # self.decoder = tf.keras.layers.LeakyReLU(alpha=0.3)                                                                                                                                 (self.decoder)
        # self.decoder = tf.keras.layers.Dense(decoded_size * 1,  activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))  (self.decoder)
        
        # self.normals = tf.keras.layers.LayerNormalization(name="out"+str(0))     (self.decoder)

        # self.last = tf.keras.layers.Add()([self.normalso[i] for i in range(num_reps)])
        # outputs = [self.normalso[i] for i in range(num_reps)]
        # outputs.append(self.last)

        self.autoenco = GradModel2(self.model_input, [self.decoder, self.encoder])
        self.autoenco.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.005, amsgrad=True), run_eagerly=False)
        self.autoenco.summary()
        
        # self.encoder_1 = tf.keras.layers.Dense(1,            activation=tf.nn.relu)(self.model_input)
        # self.decoder_1 = tf.keras.layers.Dense(decoded_size, activation=tf.nn.relu)(self.encoder_1)
        # self.normal  = tf.keras.layers.LayerNormalization()(self.decoder_1)

        # self.adder     = tf.keras.layers.Add()(self.decoder_1)

        # self.enc_e   = tf.keras.layers.Dense(decoded_size, activation=tf.nn.relu)(self.model_input)
        # self.dec_e   = tf.keras.layers.Dense(decoded_size, activation=tf.nn.relu)(self.enc_e)
        # self.nor_e   = tf.keras.layers.LayerNormalization()(self.dec_e)

        # self.dno_autoencoder = GradModel2(self.model_input, self.normal)
        # self.err_autoencoder = GradModel2(self.model_input, self.normal)

        # self.dno_autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.01, amsgrad=True), run_eagerly=False)
        # self.err_autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.01, amsgrad=True), run_eagerly=False)

        return self.autoenco, self.autoenco

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
        # b = self.err_autoencoder.predict(abs(snapshot.T-a))

        print(e)
        # print(network.layers[1].get_weights())
        # exit()

        return a.T

    def train_network(self, model, input_data, grad_data, num_files, epochs=1):
        # Train the model
        model.grads = grad_data
        model.fit(
            input_data.T, grad_data.T,
            epochs=epochs,
            batch_size=1,
            # shuffle=True,
            # validation_data=(valid_dataset, valid_dataset),
        )

        # input_data_p = abs(input_data.T - model.predict(input_data.T))

        # self.err_autoencoder.fit(
        #     input_data_p, grad_data.T,
        #     epochs=epochs,
        #     batch_size=1,
        #     # shuffle=True,
        #     # validation_data=(valid_dataset, valid_dataset),
        # )

    def calculate_gradients():
        return None

    def compute_full_gradient():
        return None