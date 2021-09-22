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

def calculate_clusters(input):
    data_rows = input.shape[0]
    data_cols = input.shape[1]

    num_clusters=3
    num_cluster_col=2
    with contextlib.redirect_stdout(None):
        cluster_bases, kmeans_object = clustering.calcualte_snapshots_with_columns(
            snapshot_matrix=input,
            number_of_clusters=num_clusters,
            number_of_columns_in_the_basis=num_cluster_col
        )

    print(" -> Generated {} cluster bases with shapes:".format(len(cluster_bases)))
    for base in cluster_bases:
        print(base, "-->", cluster_bases[base].shape)

    q = {}
    s = {}
    total_respresentations = 0
    for i in range(num_clusters):
        s[i] = input[:,kmeans_object.labels_==i]
        q[i] = cluster_bases[i].T @ input[:,kmeans_object.labels_==i]
        total_respresentations += np.shape(q[i])[1]

    print("Total number of representations:", total_respresentations)

    sc = np.empty(shape=(data_rows,0))
    qc = np.empty(shape=(num_cluster_col,0))

    for i in range(num_clusters):
        sc = np.concatenate((sc, s[i]), axis=1)
        qc = np.concatenate((qc, q[i]), axis=1)

    return sc, qc

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
        def train_step(self, data):
            x, y = data

            with tf.GradientTape(persistent=True) as tape:
                # Create tensor to calculate the model gradient
                x_tensor = tf.convert_to_tensor(x, dtype='float32')
                tape.watch(x_tensor)

                # Feed forward
                y_pred = self(x_tensor, training=True)
                dy_dx = tape.gradient(y_pred, x_tensor)

                # Continue the normal training loop
                loss_value = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            # Calculate the loss gradient
            gradients = tape.gradient(loss_value, self.trainable_variables)

            # Delete the persisten tape
            del tape
            
            # Update weights
            # Temporaly comment this because I know training will only make it worse
            # self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # Update metrics (includes the metric that tracks the loss)
            self.compiled_metrics.update_state(y, y_pred)

            # Return a dict mapping metric names to current value
            return {m.name: m.result() for m in self.metrics}

    full_dimension = 10

    matrix_input = tf.random.normal(shape=(full_dimension, 5), dtype='float64')
    matrix_input = matrix_input.numpy()

    U,S,V = np.linalg.svd(matrix_input)

    reduced_dimension = len(S)

    print("\n\tU:\n", U)
    print("\n\tS:\n", S)
    
    phi = U[:,0:reduced_dimension]

    print("\n\tPhi:\n", phi)

    print("\nPreparing Model:\n")

    # Im using linear because weights can be negative and I will set optimal values (no train)
    encoder_input  = layers.Input(shape=matrix_input.shape[0], dtype='float64', name="encoder_input")
    encoder_output = layers.Dense(reduced_dimension, activation='linear', dtype='float64', name="encoder_layer_1")(encoder_input)

    decoder_input = encoder_output

    decoder_output = layers.Dense(full_dimension,    activation='linear', dtype='float64', name="decoder_layer_1")(decoder_input)

    autoencoder = LossGradientModel(encoder_input, decoder_output, name="autoencoder")
    autoencoder.compile(loss=tf_loss, optimizer=tf.keras.optimizers.Adam(lr=0.0001, amsgrad=False))
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
    autoencoder.layers[1].set_weights([phi,   np.zeros(shape=weights_l0[1].shape)])
    autoencoder.layers[2].set_weights([phi.T, np.zeros(shape=weights_l1[1].shape)])

    print("\nPreparing datasets")

    train_dataset = matrix_input.T
    valid_dataset = matrix_input.T

    print("\n\t Train dataset shape:\n\t\t", train_dataset.shape)

    print("\n''Fake'' training phase")
    autoencoder.fit(
        train_dataset, train_dataset,
        epochs=10,
        batch_size=1,
        shuffle=True,
        validation_data=(valid_dataset, valid_dataset),
    )

    print("\nPredicting the input:")
    matrix_predicted = autoencoder.predict(matrix_input.T)
    matrix_predicted = matrix_predicted.T

    print("\n\tNorm error:\n")
    print("\t\t", np.linalg.norm(matrix_predicted-matrix_input)/np.linalg.norm(matrix_input))


    exit()


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

    print("\n =========================================================== ")
    print("\n === Executing Custom Loss Example...                    === ")
    print("\n =========================================================== ")
    example_tf()

    exit(0)

    #####################
    
    data_inputs = [
        "hdf5_output/result_30.h5",
    ]

    input = tf.random.normal(shape=(5, 5))

    print("\n =========================================================== ")
    print("\n === Calculating Cluster Bases                           === ")
    print("\n =========================================================== ")
    input, cluster = calculate_clusters(input)

    print("\n =========================================================== ")
    print("\n === Calculating Randomized Singular Value Decomposition === ")
    print("\n =========================================================== ")
    U, sigma, phi, error = calculate_svd(input)

    ######################

    ######################

    input = tf.random.normal(shape=(2, 1))
    enc_input = tf.Variable(input)

    W = tf.random.normal(shape=(1, 2))
    benc = tf.random.normal(shape=(1, 1))

    with tf.GradientTape() as tape:
        # tape.watch(W)
        # tape.watch(b)

        print("W:", W)
        print("b:", benc)
        print("Relu in:", W@enc_input + benc)

        encoder_out = tf.nn.relu(W@enc_input + benc)

        print("Relu:", encoder_out)

        # decoder_in = encoder_out
        # decoder = W.T@decoder_in + bdec

        grad_encoder = tape.gradient(encoder_out, enc_input)

        # loss = norm(decoder_out - encoder_in) + norm(grad_encoder - G)
        # Dloss/Dw   ,   Dloss, benc

        print("Gradi", grad_encoder)

    ######################

    # print("input shape:", np.shape(input))
    # predicted = encoder(input)

    # print("Predicted:", predicted)