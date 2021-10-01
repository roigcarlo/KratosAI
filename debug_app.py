import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import math

import contextlib

import h5py
import numpy as np

import keras
import tensorflow as tf

from keras import layers
from itertools import repeat

import kratos_io
import clustering
import networks.debug_network as debug_network

import matplotlib.pyplot as plt

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as romapp

from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

class DenseLayer:
    """ Simple class to mimic the tt.layers.dense """
    def __init__(self, w_shape, b_shape, activator):
        self._w = tf.random.normal(shape=w_shape, dtype="float64")
        self._b = tf.zeros(shape=b_shape, dtype="float64")
        self._activator = activator

    def SetW(self, w):
        self._w = w.copy()

    def SetB(self, b):
        self._b = b.copy()

    def GetW(self):
        return self._w

    def GetB(self):
        return self._b

    def Execute(self, input):
        # print("======")
        print("W:", self._w.shape)
        print("I:", input.shape)
        # print("@:", self._w@input)
        # print("======")

        # return self._activator(self._w@input + self._b)
        return self._w@input

def calculate_svd(input):
    with contextlib.redirect_stdout(None):
        U,sigma,phi,error = RandomizedSingularValueDecomposition().Calculate(input, 1e-6)
    return U,sigma,phi,error

def calculate_clusters_with_columns(input, num_clusters, num_cluster_col):
    data_rows = input.shape[0]
    data_cols = input.shape[1]

    with contextlib.redirect_stdout(None):
        cluster_bases, kmeans_object = clustering.calcualte_snapshots_with_columns(
            snapshot_matrix=input,
            number_of_clusters=num_clusters,
            number_of_columns_in_the_basis=num_cluster_col
        )

    return cluster_bases, kmeans_object

def calculate_clusters(input, num_clusters, num_cluster_col):
    data_rows = input.shape[0]
    data_cols = input.shape[1]

    with contextlib.redirect_stdout(None):
        cluster_bases, kmeans_object = clustering.calcualte_snapshots_with_columns(
            snapshot_matrix=input,
            number_of_clusters=num_clusters,
            number_of_columns_in_the_basis=num_cluster_col
        )

    # print(" -> Generated {} cluster bases with shapes:".format(len(cluster_bases)))
    # for base in cluster_bases:
    #     print(base, "-->", cluster_bases[base].shape)

    s = {}
    q = {}
    g = {}
    total_respresentations = 0
    for i in range(num_clusters):
        s[i] = input[:,kmeans_object.labels_==i]
        q[i] = cluster_bases[i].T @ input[:,kmeans_object.labels_==i] 
        g[i] = cluster_bases[i] @ cluster_bases[i].T  
        total_respresentations += np.shape(q[i])[1]

    sc = np.empty(shape=(data_rows,0))
    qc = np.empty(shape=(num_cluster_col,0))
    gc = np.empty(shape=(data_rows, 0))

    for i in range(num_clusters):
        sc = np.concatenate((sc, s[i]), axis=1)
        qc = np.concatenate((qc, q[i]), axis=1)
        gc = np.concatenate((gc, g[i]), axis=1)

    return sc, qc, gc

def example_w():
    a = np.array([1,2,3])

    A = np.zeros((3,4))
    A[:,0] = a 
    A[:,1] = 2*a 
    A[:,2] = 5*a
    A[:,3] = 3*a

    U,S,V = np.linalg.svd(A)

    print("\n\tA:\n", A)

    print("\n\tU:\n", U)
    print("\n\tS:\n", S)
    print("\n\tV:\n", V)

    phi = U[:,0].reshape((3,1))
    print("\n\tPhi:\n", phi)
    print("\n\tphi@phi.T:\n", phi@phi.T)
    print("\n\tphi@phi.T@A - A:\n", phi@phi.T@A - A)

    # # Version Ric: con derivada de todo el autoencoder con respeto al input
    # autoencoder = W@W.T@input
    # loss = norm(autoencoder(input)-input) + h*norm((grad(autoencoder,input) - phi@phi.T))

    # # Version Charlie: con la derivada solo del encoder
    # encoder = W.T@input_enc
    # decoder = W@input_dec
    # grad_dec = grad(decoder, input_dec)

    # loss = norm(decoder(encoder(input))-input) + h*norm((grad_dec - phi ))

def example_raw():
    input = tf.random.normal(shape=(2, 1))
    enc_input = tf.Variable(input)

    W = tf.random.normal(shape=(1, 2))
    benc = tf.random.normal(shape=(1, 1))

    with tf.GradientTape() as tape:
        print("\n\tW:\n", W)
        print("\n\tb:\n", benc)
        print("\n\tRelu in:\n", W@enc_input + benc)

        encoder_out = tf.nn.relu(W@enc_input + benc)

        print("\n\tRelu out:\n", encoder_out)
        grad_encoder = tape.gradient(encoder_out, enc_input)

        print("\n\tGradi:\n", grad_encoder)

def example_mockup():
    matrix_input = tf.random.normal(shape=(5, 1))

    U,S,V = np.linalg.svd(matrix_input)
    print("\n\tU:\n", U)

    encoder_input = tf.Variable(matrix_input)
    
    encoder = DenseLayer(w_shape=(1, 5), b_shape=(1, 1), activator=tf.nn.relu)
    decoder = DenseLayer(w_shape=(5, 1), b_shape=(5, 1), activator=tf.nn.relu)

    # Make the W simetric and use U.
    encoder.SetW(U[:,0].reshape(1,5))
    decoder.SetW(U[:,0].reshape(1,5).T)

    with tf.GradientTape() as tape:

        encoder_output = encoder.Execute(encoder_input)
        decoder_input  = encoder_output
        decoder_output = decoder.Execute(decoder_input)

        grandient = tape.gradient(decoder_output, encoder_input)

        print("\n\tGradi:\n", grandient)
        print("\n\tEncoder in:\n", encoder_input)
        print("\n\tDecoder out:\n", decoder_output)

        print("\n\tPhi@Phi.T@encoder_input", U[:,0].reshape(5,1) @ U[:,0].reshape(5,1).T @ matrix_input)

def example_custom_loss():
    class MyLossHelper:
        def __init__(self, gradient):
            self.analytical_gradient = gradient
        
        def compute_loss(y_true, y_pred):
            # (y_true - y_pred) ** 2 + (self.analytical_gradient - ???) ** 2 
            pass
            
    matrix_input = tf.math.abs(tf.random.normal(shape=(5, 50)))

    matrix_input = np.zeros((5,50))

    v1 = np.random.rand(5,1)
    v2 = np.random.rand(5,1)
    random_scalars1 = np.random.rand(matrix_input.shape[1],1)
    random_scalars2 = np.random.rand(matrix_input.shape[1],1)
    for i in range(matrix_input.shape[1]):
        matrix_input[:,i] = v1[:,0] * random_scalars1[i,0] + v2[:,0] * random_scalars2[i,0]

    input, cluster = calculate_clusters(matrix_input)

    print("\n\tSize of input  : \t", np.shape(input))
    print("\n\tSize of cluster: \t", np.shape(cluster))

    encoder_input = np.concatenate((input, cluster), axis=0)

    print("\n\tSize of encoder_input: \t", np.shape(encoder_input))

    U,S,V = np.linalg.svd(matrix_input)
    print("\n\tU:\n", U)
    print("\n\tS:\n", S)

    tf_encoder_input = tf.Variable(encoder_input)
    
    encoder = DenseLayer(w_shape=(2, 7), b_shape=(2, 1), activator=tf.nn.relu)
    decoder = DenseLayer(w_shape=(7, 2), b_shape=(7, 1), activator=tf.nn.relu)

    # Make the W simetric and use U.
    encoder_weights = U[:,0:2]
    # decoder_weights = U[:,0].reshape(1,5).T
    print("\n\tencoder_weights:\n",encoder_weights.shape, encoder_weights)

    encoder_weights = np.concatenate((encoder_weights, np.asarray([[0.0, 0.0], [0.0, 0.0]])), axis=0).T
    decoder_weights = encoder_weights.T
    print("\n\tencoder_weights:\n",encoder_weights)

    encoder.SetW(encoder_weights)
    decoder.SetW(decoder_weights)

    ei_shape = encoder_input.shape
    print("\n\tei_shape:\n", ei_shape)

    for snap in range(ei_shape[1]):

        print("\n ============ Iteration:", snap, " ============ \n")
        encoder_input_n = tf.Variable(np.asarray([encoder_input[:,snap]]).T)
        print("\n\tInput for iteration", snap, ":\n", encoder_input_n)

        with tf.GradientTape() as tape:

            encoder_output = encoder.Execute(encoder_input_n)
            decoder_input  = encoder_output
            decoder_output = decoder.Execute(decoder_input)

            grandient = tape.gradient(decoder_output, encoder_input_n)

            print("\n\t\tGradi:      \n", grandient)
            print("\n\t\tEncoder in: \n", encoder_input_n)
            print("\n\t\tDecoder out:\n", decoder_output)

            # print("\n\tPhi@Phi.T@encoder_input", U[:,0].reshape(7,1) @ U[:,0].reshape(7,1).T @ matrix_input)

def example_tf():

    # Custom loss function
    def tf_loss(y_true, y_pred):
        y_diff = y_pred - y_true
        y_diff = y_diff ** 2

        return y_diff

    # Inherit from Keral model so we can use the existing model.fit
    class LossGradientModel(keras.Model):
        def save_model_gradient(self, model_gradient):
            self.model_gradient = model_gradient

        def save_model_weights(self, model_weights):
            self.model_weights = model_weights

        def tf_gradient_loss(self, y_true, y_pred, m_grad, m_pred):
            y_diff = y_pred - y_true
            y_diff = y_diff ** 2

            m_diff = m_pred - m_grad
            m_diff = m_diff ** 2

            return tf.math.reduce_mean(y_diff) + k * tf.math.reduce_mean(m_diff)

        def train_step(self, data):
            x, y = data

            with tf.GradientTape(persistent=True) as tape:
                # Create tensor to calculate the model gradient
                x_tensor = tf.convert_to_tensor(x, dtype='float32')
                # x_tensor = tf.cast(x, dtype='float64')
                
                tape.watch(x_tensor)

                # Feed forward
                W  = self.layers[1].get_weights()
                WT = self.layers[2].get_weights()

                Wt  = tf.Variable(W[0],  dtype='float64')
                WTt = tf.Variable(WT[0], dtype='float64')

                # tape.watch(Wt)
                # tape.watch(WTt)

                #y_pred = Wt@WTt@tf.transpose(x_tensor)
                y_pred = self(x_tensor, training=True)

                g = []
                for i in range(y_pred.shape[1]):
                    gi = tape.gradient(y_pred[0][i], x_tensor)
                    g.append(gi)
                # dy_dx = dy_dW @ dy_dWT
                
                # xy_xx = self.ideal_gradient
                # yy_yx = Wt @ WTt
                # print(dy_dW, dy_dWT)
                print(g)

                # Continue the normal training loop
                # loss_value = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
                loss_value = self.tf_gradient_loss(x_tensor, y_pred, xy_xx, yy_yx)
                # loss_value = tf.constant(0)

            self.save_model_gradient(g)
            # self.save_model_weights(yy_yx)

            # Calculate the loss gradient
            gradients = tape.gradient(loss_value, self.trainable_variables)

            # Delete the persisten tape
            del tape
            
            # Update weights
            # Temporaly comment this because I know training will only make it worse
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(y, y_pred)

            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}

    full_dimension  = 10
    num_snapshots   = 20
    n_columns       = 10

    # Fix the seed otherwise debug is painfull
    tf.random.set_seed(1)

    v1_input = tf.random.normal(shape=(full_dimension, 1), dtype='float64')
    v2_input = tf.random.normal(shape=(full_dimension, 1), dtype='float64')

    v1_input = v1_input.numpy()
    v2_input = v2_input.numpy()

    lambdas = tf.random.normal(shape=(n_columns, 2), dtype='float64')

    lambdas = lambdas.numpy()

    matrix_input = np.zeros(shape=(full_dimension, n_columns))
    for i in range(n_columns):
        matrix_input[:,i] = (v1_input * lambdas[i, 0] + v2_input * lambdas[i, 1]).reshape(full_dimension)

    # matrix_input = matrix_input.numpy()
    # matrix_disto = matrix_disto.numpy()

    U,S,V = np.linalg.svd(matrix_input)

    # We can trim this to make the results worst
    reduced_dimension = 2# len(S)

    print("\n\tU:\n", U)
    print("\n\tS:\n", S)
    
    phi = U[:,0:reduced_dimension]

    print("\n\tPhi Shap:\n", phi.shape)
    print("\n\tPhi:\n", phi)

    print("\nPreparing Model:\n")

    # Im using linear because weights can be negative and I will set optimal values (no train)
    encoder_input  = layers.Input(shape=matrix_input.shape[0], dtype='float64', name="encoder_input")
    encoder_output = layers.Dense(reduced_dimension, activation='linear', dtype='float64', name="encoder_layer_1")(encoder_input)

    decoder_input = encoder_output

    decoder_output = layers.Dense(full_dimension,    activation='linear', dtype='float64', name="decoder_layer_1")(decoder_input)

    autoencoder = LossGradientModel(encoder_input, decoder_output, name="autoencoder")

    # Needs to run eagerly for debug
    autoencoder.compile(loss=tf_loss, optimizer=tf.keras.optimizers.Adam(lr=0.0001, amsgrad=False), run_eagerly=True)
    autoencoder.summary()

    print("\nAssigning optimal weights:\n")

    # autoencoder.layers[0] is the input layer
    weights_l0 = autoencoder.layers[1].get_weights()
    weights_l1 = autoencoder.layers[2].get_weights()

    print("\n\tLayer Weights")
    print("\t\tL0: W  :", weights_l0[0].shape, "B:", weights_l0[1].shape)
    print("\t\tL1: W  :", weights_l1[0].shape, "B:", weights_l1[1].shape)

    print("\n\tOptimal Weights")
    print("\t\tL0: φ  :", phi.shape)
    print("\t\tL1: φ.T:", phi.T.shape)

    if weights_l0[0].shape != phi.shape or weights_l1[0].shape != phi.T.shape:
        print("Error: optimal weights mismatch")
        exit(1)

    # Bias is set to 0 because network is optimal
    noise_factor = 1e-5
    autoencoder.layers[1].set_weights([phi   + tf.random.normal(phi.shape)   * noise_factor, np.zeros(shape=weights_l0[1].shape)])
    autoencoder.layers[2].set_weights([phi.T + tf.random.normal(phi.T.shape) * noise_factor, np.zeros(shape=weights_l1[1].shape)])

    print("\n\tClusters:")
    reorder_inputs, clusters, grads = calculate_clusters(matrix_input, 1, reduced_dimension)
    print("\n\tC:", reorder_inputs.shape, reorder_inputs[0])
    print("\n\tK:", clusters.shape,  clusters[0])
    print("\n\tG:", grads.shape,   grads)
    autoencoder.ideal_gradient = grads

    print("\nPreparing datasets:")

    train_dataset = reorder_inputs.T
    valid_dataset = reorder_inputs.T

    target_dataset = reorder_inputs.T

    print("\n\t Train dataset shape:\n\t\t", train_dataset.shape)

    print("\n''Fake'' training phase")
    autoencoder.fit(
        train_dataset, train_dataset,
        epochs=500,
        batch_size=1,
        shuffle=True,
        validation_data=(valid_dataset, valid_dataset),
    )

    # print("\n\tModel Gradient:\n\t\t", autoencoder.model_gradient)
    print("\n\tModel Weights:\n\t\t", autoencoder.model_weights)

    print("\nPredicting the input:")
    matrix_predicted = autoencoder.predict(reorder_inputs.T)
    matrix_predicted = matrix_predicted.T

    print("\nPrediction of column 0:")
    print("\n\tOrg:\n\t\t", reorder_inputs.T[0])
    print("\n\tPre:\n\t\t", matrix_predicted.T[0])

    print("\n\tNorm error:\n")
    print("\t\t", np.linalg.norm(matrix_predicted-reorder_inputs)/np.linalg.norm(reorder_inputs))

    exit()

def example_custom_grad():

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
            gradient = tf.Variable(grad_vals, dtype="float32") @ self.weights[0] + self.weights[1]
        else:
            gradient = tf.Variable(grad_vals, dtype="float32") @ self.weights[0]
        
        return gradient

    def ReluGrad(self, outputs):
        grad_vals = np.zeros((outputs.shape[1], outputs.shape[1]))
        for i in range(outputs.shape[1]):
            if outputs[0,i] > 0:
                grad_vals[i,i] = 1.0

        if self.use_bias:
            gradient = tf.Variable(grad_vals, dtype="float32") @ self.weights[0] + self.weights[1]
        else:
            gradient = tf.Variable(grad_vals, dtype="float32") @ self.weights[0]

        return gradient

    def CallWithGrads(self, inputs):
        outputs = self.call_base(inputs)
        self.gradient = self.call_grad(outputs)
        return outputs

    # Create a custom Model:
    loss_tracker = keras.metrics.Mean(name="loss")
    mae_metric = keras.metrics.MeanAbsoluteError(name="mae")

    class GradModel(keras.Sequential):
        def grad_loss(self, y_true, y_pred):
            return (y_true - y_pred) ** 2

        def train_step(self, data):
            x, y = data

            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = self(x, training=True)

                # Compute our own loss
                # loss = keras.losses.mean_squared_error(y, y_pred)
                loss = self.grad_loss(a, self.layers[0].weights[0])

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Compute our own metrics
            loss_tracker.update_state(loss)
            mae_metric.update_state(y, y_pred)
            return {"loss": loss_tracker.result(), "mae": mae_metric.result()}

        @property
        def metrics(self):
            # We list our `Metric` objects here so that `reset_states()` can be
            # called automatically at the start of each epoch
            # or at the start of `evaluate()`.
            # If you don't implement this property, you have to call
            # `reset_states()` yourself at the time of your choosing.
            return [loss_tracker, mae_metric]

    # Create GradientLayers Protos
    LineLayerProto = GradExtender(keras.layers.Dense, CallWithGrads, LineGrad)
    ReLULayerProto = GradExtender(keras.layers.Dense, CallWithGrads, ReluGrad)

    ### Sample Run ###

    num_samples = 100

    a     = np.array([[1.0, 2.0], [3.0, 4.0]])
    input = tf.random.normal(shape=(num_samples, 2)) 

    exact_result = input @ a

    model = GradModel(
        [
            LineLayerProto(2, activation="relu", name="LinearDense", use_bias=True)
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=False),
        run_eagerly=True
    )

    model.fit(
        input, exact_result,
        epochs=2,
        batch_size=1
    )

    print(model.layers[0].weights)
    
def example_custom_grad_multi_layer():

    def GenInputAsDependentColumns():
        # Fix the seed otherwise debug is painfull
        tf.random.set_seed(1)

        v1_input = tf.random.normal(shape=(full_dimension, 1), dtype='float64')
        v2_input = tf.random.normal(shape=(full_dimension, 1), dtype='float64')

        v1_input = v1_input.numpy()
        v2_input = v2_input.numpy()

        lambdas = tf.random.normal(shape=(n_columns, 2), dtype='float64')

        lambdas = lambdas.numpy()

        matrix_input = np.zeros(shape=(full_dimension, n_columns))
        for i in range(n_columns):
            matrix_input[:,i] = (v1_input * lambdas[i, 0] + v2_input * lambdas[i, 1]).reshape(full_dimension)

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

        # This is ***possibly*** wrong and was working because the 2x2 example was symetric
        # if self.use_bias:
        #     gradient = tf.Variable(grad_vals, dtype="float32") @ self.weights[0] + self.weights[1]
        # else:
        #     gradient = tf.Variable(grad_vals, dtype="float32") @ self.weights[0]

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

        # This is ***possibly*** wrong and was working because the 2x2 example was symetric
        # if self.use_bias:
        #     gradient = tf.Variable(grad_vals, dtype="float32") @ self.weights[0] + self.weights[1]
        # else:
        #     gradient = tf.Variable(grad_vals, dtype="float32") @ self.weights[0]

        if self.use_bias:
            gradient = self.weights[0] @ tf.Variable(grad_vals, dtype="float32") + self.weights[1]
        else:
            gradient = self.weights[0] @ tf.Variable(grad_vals, dtype="float32")

        return gradient

    def CallWithGrads(self, inputs):
        outputs = self.call_base(inputs)
        self.gradient = self.call_grad(outputs)
        return outputs

    # Create a custom Model:
    loss_tracker = keras.metrics.Mean(name="loss")
    mse_metric = keras.metrics.MeanSquaredError(name="mse")

    class GradModel(keras.Sequential):
        def diff_loss(self, y_true, y_pred):
            return (y_true - y_pred) ** 2

        def grad_loss(self, m_true, m_pred):
            return (m_true - m_pred) ** 2

        def combind_loss(self, y_true, y_pred, g_true, g_pred):
            diff_loss = self.diff_loss(y_true, y_pred)
            grad_loss = self.grad_loss(g_true, g_pred)

            return tf.math.reduce_mean(diff_loss) + tf.math.reduce_mean(grad_loss)

        def train_step(self, data):
            x, y = data

            with tf.GradientTape(persistent=True) as tape:
                # Forward pass
                y_pred = self(x, training=True)

                m_grad_pred = self.layers[0].gradient
                for i in range(1, len(self.layers)):
                    m_grad_pred = m_grad_pred @ self.layers[i].gradient

                # # Compute our own loss
                # loss_d = self.diff_loss(y, y_pred)
                # loss_m = self.grad_loss(m_grad, m_grad_pred)

                # Compute our own loss
                loss = self.grad_loss(m_grad, m_grad_pred)
                # loss = self.combind_loss(y, y_pred, m_grad, m_grad_pred)

            # # Compute gradients
            # trainable_vars = self.trainable_variables
            # gradients_d = tape.gradient(loss_d, trainable_vars)
            # gradients_m = tape.gradient(loss_m, trainable_vars)

            # # Update weights
            # self.optimizer.apply_gradients(zip(gradients_d, trainable_vars))
            # self.optimizer.apply_gradients(zip(gradients_m, trainable_vars))

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Compute our own metrics
            # loss_tracker.update_state(loss_d, loss_m)
            loss_tracker.update_state(loss)

            # Update metrics
            # mse_metric.update_state(y, y_pred)
            mse_metric.update_state(m_grad, m_grad_pred)
            return {"loss": loss_tracker.result(), "mse": mse_metric.result()}

        @property
        def metrics(self):
            # We list our `Metric` objects here so that `reset_states()` can be
            # called automatically at the start of each epoch
            # or at the start of `evaluate()`.
            # If you don't implement this property, you have to call
            # `reset_states()` yourself at the time of your choosing.
            return [loss_tracker, mse_metric]

    # Create GradientLayers Protos
    LineLayerProto = GradExtender(keras.layers.Dense, CallWithGrads, LineGrad)
    ReLULayerProto = GradExtender(keras.layers.Dense, CallWithGrads, ReluGrad)

    ### Sample Run ###

    num_vars = 1024
    num_samples = 100

    full_size = num_vars
    enc_size = 10

    normalized1, norm1 = tf.linalg.normalize(tf.abs(tf.random.normal(shape=(num_vars, enc_size))))
    normalized2, norm2 = tf.linalg.normalize(tf.abs(tf.random.normal(shape=(enc_size, num_vars))))

    ideal_layer_weights = [
          normalized1
        , normalized2
    ]

    m_grad = ideal_layer_weights[0]
    for l in ideal_layer_weights[1:]:
        m_grad = m_grad @ l

    input, norm_input = tf.linalg.normalize(tf.abs(tf.random.normal(shape=(num_samples, num_vars))))

    print("Input Shape:", input.shape)

    exact_result = input
    for l in ideal_layer_weights:
        exact_result = exact_result @ l

    model = GradModel(
        [
              LineLayerProto(enc_size,  activation="linear", name="LinearDense1", use_bias=False)
            , LineLayerProto(full_size, activation="linear", name="LinearDense2", use_bias=False)
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True),
        run_eagerly=True
    )

    model.fit(
        input, exact_result,
        epochs=5,
        batch_size=1
    )

    print("\nGradients:")
    m_grad_pred = model.layers[0].weights[0]
    print("\nLayer0:", model.layers[0].weights[0])
    for i in range(1, len(model.layers)):
        print("\nLayer"+str(i)+";", model.layers[i].weights[0])
        m_grad_pred = m_grad_pred @ model.layers[i].weights[0]
    print("\nIdeal:", m_grad, "\nNet  :", m_grad_pred)

    print("\nResults:")
    expect = exact_result[0]
    result = model(np.array([input[0]]))
    print("\nCalc:", expect, "\nPred:", result)
    print("\nCompression:", (1-enc_size/num_vars)*100, "%\tNorm Error:", np.linalg.norm(expect-result)/np.linalg.norm(expect))

if __name__ == "__main__":

    # print("\n =========================================================== ")
    # print("\n === Executing Phi Example...                            === ")
    # print("\n =========================================================== ")
    # example_w()

    # print("\n =========================================================== ")
    # print("\n === Executing Tape Example...                           === ")
    # print("\n =========================================================== ")
    # example_raw()

    # print("\n =========================================================== ")
    # print("\n === Executing Mockup Example...                         === ")
    # print("\n =========================================================== ")
    # example_mockup()

    # print("\n =========================================================== ")
    # print("\n === Executing Custom Loss Example...                    === ")
    # print("\n =========================================================== ")
    # example_tf()

    # print("\n =========================================================== ")
    # print("\n === Executing Custom Grad Example...                    === ")
    # print("\n =========================================================== ")
    # example_custom_grad()

    print("\n =========================================================== ")
    print("\n === Executing Custom Grad Example...                    === ")
    print("\n =========================================================== ")
    example_custom_grad_multi_layer()


    exit(0)