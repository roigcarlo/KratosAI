from __future__ import print_function, absolute_import, division #makes KratosMultiphysics backward compatible with python 2.6 and 2.7

import json
import math

import h5py
import numpy as np

import keras
import tensorflow as tf

from keras import layers
from itertools import repeat

import kratos_io
import clustering
import shallow_autoencoder

import matplotlib.pyplot as plt

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as romapp

from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

def print_gpu_info():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

if __name__ == "__main__":

    save_model = True
    load_model = False
    
    data_inputs = [
        "hdf5_output/result_30.h5",
        "hdf5_output/result_35.h5",
        "hdf5_output/result_40.h5",
        "hdf5_output/result_45.h5",
        "hdf5_output/result_50.h5",
        "hdf5_output/result_55.h5",
        "hdf5_output/result_60.h5",
        "hdf5_output/result_65.h5",
        "hdf5_output/result_70.h5",
        "hdf5_output/result_75.h5",
        "hdf5_output/result_80.h5",
        "hdf5_output/result_85.h5",
        "hdf5_output/result_90.h5",
        "hdf5_output/result_95.h5",
        "hdf5_output/result_100.h5",
    ]

    kratos_network = shallow_autoencoder.ShallowAutoencoder()
    S = kratos_io.build_snapshot_grid(
        data_inputs, 
        [
            # "PRESSURE", 
            "VELOCITY"
        ]
    )

    kratos_io.print_npy_snapshot(S, True)
    exit()

    print("S", S.shape)

    U,sigma,_,error = RandomizedSingularValueDecomposition().Calculate(S,1e-6)

    print("U", U.shape)

    SReduced = U.T@S

    print("SReduced", SReduced.shape)

    def custom_loss(y_true, y_pred):
        y_diff = y_true-y_pred
        y_diff = y_diff ** 2

        return  y_diff

    load_model = False
    train_model = True
    save_model = True

    if load_model:
        autoencoder = tf.keras.models.load_model(kratos_network.model_name, custom_objects={"custom_loss":custom_loss})
    else:
        autoencoder = kratos_network.define_network(SReduced, custom_loss )

    encoder = autoencoder.layers[1]
    decoder = autoencoder.layers[2]

    encoder.summary()
    decoder.summary()
    autoencoder.summary()

    kratos_network.calculate_data_limits(SReduced)

    if train_model:
        kratos_network.train_network(autoencoder, SReduced, len(data_inputs))

    if save_model:
        autoencoder.save(kratos_network.model_name)

    # Obtain Clusters
    cluster_bases = clustering.calcualte_snapshots(
        snapshot_matrix=S,
        number_of_clusters=5
    )
        
    # Obtain Decoder/Encoder Derivatives.
    snapshot_index = 5

    all_gradients = kratos_network.calculate_gradients(
        U.T@S[:,snapshot_index], 
        encoder, decoder, custom_loss,
        decoder.trainable_variables
    )

    full_gradient = kratos_network.compute_full_gradient(autoencoder, all_gradients)

    SEncoded = kratos_network.encode_snapshot(encoder, SReduced) # This is q,  or g(u)
    SDecoded = kratos_network.decode_snapshot(decoder, SEncoded) # This is u', or f(q), or f(g(u)) 

    # Verify that results are correct
    SPredict = kratos_network.predict_snapshot(autoencoder, SReduced) # This is u', or f(q), or f(g(u)) 

    # This should be 0.
    if np.count_nonzero(SDecoded - SPredict):
        print("[ERROR]: Autoencoder != Decoder·Encoder")

    SP = U@SDecoded

    # kratos_network.check_gradient(SEncoded, SDecoded)

    # print("dq_vector:", dq_vector) # This is \nabla{q} for the encoder.

    # exit(0)

    # SP = np.zeros(S.shape)

    # for i in range(0, S.shape[1]):
    #     col = S[:,i]
    #     col_reduced = U.T@col
    #     col_pred_r = KratosTrainer.predict_vector(autoencoder, col_reduced)
    #     col_pred = U@col_pred_r
    #     SP[:,i] = col_pred[:,0]

    print("S  Shape:",  S.shape)
    print("SP Shape:", SP.shape)
    print(np.linalg.norm(SP-S)/np.linalg.norm(S))

    current_model = KMP.Model()
    model_part = current_model.CreateModelPart("main_model_part")

    kratos_io.create_out_mdpa(model_part, "GidExampleSwaped")
    kratos_io.print_results_to_gid(model_part, S.T, SP.T)
