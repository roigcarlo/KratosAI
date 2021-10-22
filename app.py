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
import networks.shallow_autoencoder as shallow_autoencoder
import networks.cluster_autoencoder as cluster_autoencoder
import networks.gradient_shallow    as gradient_shallow_ae

import matplotlib.pyplot as plt

import KratosMultiphysics as KMP
import KratosMultiphysics.RomApplication as romapp

from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition

def print_gpu_info():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

def custom_loss(y_true, y_pred):
    y_diff = y_true-y_pred
    y_diff = y_diff ** 2

    return  y_diff

def custom_loss_cluster(y_true, y_pred):        
    y_diff = y_true-y_pred
    y_diff = y_diff ** 2

    return  y_diff

if __name__ == "__main__":
    
    data_inputs = [
        # "hdf5_output/result_30.h5",
        # "hdf5_output/result_35.h5",
        # "hdf5_output/result_40.h5",
        # "hdf5_output/result_45.h5",
        "hdf5_output/result_50.h5",
        # "hdf5_output/result_55.h5",
        # "hdf5_output/result_60.h5",
        # "hdf5_output/result_65.h5",
        # "hdf5_output/result_70.h5",
        # "hdf5_output/result_75.h5",
        # "hdf5_output/result_80.h5",
        # "hdf5_output/result_85.h5",
        # "hdf5_output/result_90.h5",
        # "hdf5_output/result_95.h5",
        # "hdf5_output/result_100.h5",
    ]

    config = {
        "load_model":       False,
        "train_model":      True,
        "save_model":       True,
        "print_results":    True
    }

    S = kratos_io.build_snapshot_grid(
        data_inputs, 
        [
            # "PRESSURE", 
            "VELOCITY"
        ]
    )

    kratos_network = gradient_shallow_ae.GradientShallow()
    kratos_io.print_npy_snapshot(S, True)

    print("S", S.shape)

    print("=== Calculating Randomized Singular Value Decomposition ===")
    with contextlib.redirect_stdout(None):
        U,sigma,_,error = RandomizedSingularValueDecomposition().Calculate(S,1e-6)

    print("U", U.shape)

    SReduced = U.T@S

    print("SReduced", SReduced.shape)

    data_rows = SReduced.shape[0]
    data_cols = SReduced.shape[1]

    kratos_network.calculate_data_limits(SReduced)

    # Set the properties for the clusters
    num_clusters=1      # Number of different bases chosen
    num_cluster_col=5  # If I use num_cluster_col = num_variables result should be exact.

    if config["load_model"]:
        autoencoder = tf.keras.models.load_model(kratos_network.model_name, custom_objects={"custom_loss":custom_loss})
    else:
        autoencoder = kratos_network.define_network(SReduced, custom_loss_cluster, num_cluster_col)

    # Prepare the data
    SReducedNormalized = (SReduced - kratos_network.data_min) / (kratos_network.data_max - kratos_network.data_min)

    # Obtain Clusters
    print("=== Calculating Cluster Bases ===")
    with contextlib.redirect_stdout(None):
        cluster_bases, kmeans_object = clustering.calcualte_snapshots_with_columns(
            snapshot_matrix=SReducedNormalized,
            number_of_clusters=num_clusters,
            number_of_columns_in_the_basis=num_cluster_col
        )

    print(f" -> Generated {len(cluster_bases)=} cluster bases with shapes:")
    for base in cluster_bases:
        print(f' ---> {base=} has {cluster_bases[base].shape=}')
        # print(base, "-->", cluster_bases[base].shape)

    # Obtain the reduced representations (that would be our G's)
    q, s, c = {}, {}, {}
    total_respresentations = 0

    for i in range(num_clusters):
        q[i] = cluster_bases[i].T @ SReduced[:,kmeans_object.labels_==i]
        c[i] = cluster_bases[i] @ cluster_bases[i].T
        s[i] = SReduced[:,kmeans_object.labels_==i]
        total_respresentations += np.shape(q[i])[1]
        print("Reduced representations:", np.shape(c[i]))

    print("Total number of representations:", total_respresentations)

    temp_size = num_cluster_col
    grad_size = 54
    qc = np.empty(shape=(temp_size,0))
    sc = np.empty(shape=(data_rows,0))
    cc = np.empty(shape=(grad_size,0))

    for i in range(num_clusters):
        qc = np.concatenate((qc, q[i]), axis=1)
        sc = np.concatenate((sc, s[i]), axis=1)
        cc = np.concatenate((cc, c[i]), axis=1)

    nqc = kratos_network.normalize_data(qc)
    nsc = kratos_network.normalize_data(sc)
    ncc = kratos_network.normalize_data(cc)

    print(f"{np.shape(nqc)=}")
    print(f"{np.shape(nsc)=}")
    print(f"{np.shape(ncc)=}")

    r = SReduced.T[0] @ c[0]

    print(f"Norm error using any of the {num_clusters} clusters: {np.linalg.norm(r-SReduced.T[0])/np.linalg.norm(SReduced.T[0])}")

    # autoencoder.summary()

    for layer in range(len(autoencoder.layers)):
        if hasattr(autoencoder.layers[layer], "summary"):
            print("Summary of layer", layer)
            autoencoder.layers[layer].summary()

    encoder = autoencoder.layers[0]
    decoder = autoencoder.layers[1]

    if config["train_model"]:
        autoencoder.set_m_grad(c[0])
        kratos_network.train_network(autoencoder, nsc, nqc, len(data_inputs))

    if config["save_model"]:
        autoencoder.save(kratos_network.model_name)

    # ============ This is a test ============
    # Force the autoencoder to have the wieghs
    # equal to the ones of the clusters calc
    # (the nn is sequentia so I need to run a fit step before setting the values to initialize it)

    # Without bias
    # autoencoder.layers[0].set_weights([cluster_bases[0]])
    # autoencoder.layers[1].set_weights([cluster_bases[0].T])

    # With bias
    autoencoder.layers[0].set_weights([cluster_bases[0], np.zeros(shape=autoencoder.layers[0].get_weights()[1].shape)])
    autoencoder.layers[1].set_weights([cluster_bases[0].T, np.zeros(shape=autoencoder.layers[1].get_weights()[1].shape)])

    # Retrain with the correct weights
    kratos_network.train_network(autoencoder, nsc, nqc, len(data_inputs), 50)
    # ========================================

    # # Use the network
    # SEncoded = kratos_network.encode_snapshot(encoder, SReduced)        # This is q,  or g(u)
    # SDecoded = kratos_network.decode_snapshot(decoder, SEncoded)        # This is u', or f(q), or f(g(u)) 

    # Verify that results are correct
    to_predict = np.array([nsc.T[0]])

    NSPredict = kratos_network.predict_snapshot(autoencoder, nsc)        # This is u', or f(q), or f(g(u)) 
    SPredict  = kratos_network.denormalize_data(NSPredict)

    # Compare a snap in the middle
    print(nsc.T[100])
    print(NSPredict.T[100])

    # # This should be 0. (これはゼロでなければなりません)
    # if np.count_nonzero(SDecoded - SPredict):
    #     print("[ERROR]: Autoencoder != Decoder·Encoder")
        
    # # Obtain Decoder/Encoder Derivatives.
    # snapshot_index = 5

    # all_gradients = kratos_network.calculate_gradients(
    #     U.T@S[:,snapshot_index], 
    #     encoder, decoder, custom_loss,
    #     decoder.trainable_variables
    # )

    # full_gradient = kratos_network.compute_full_gradient(autoencoder, all_gradients)

    # print("Gradient shape at index {}: {}".format(snapshot_index, full_gradient.shape))

    print("Prediction Shape ([2, ?]?):", np.shape(SPredict[0]))
    print("Prediction Shape ([ , 3]?):", np.shape(SPredict[1]))

    print("U:", U.shape)

    SP = U@(SPredict)

    print("S  Shape:",  S.shape)
    print("SP Shape:", SP.shape)

    print("SP norm error", np.linalg.norm(SP-S)/np.linalg.norm(S))
    # print("GR norm error", np.linalg.norm(autoencoder.m_grad_pred-autoencoder.m_grad)/np.linalg.norm(autoencoder.m_grad))

    print(autoencoder.m_grad_pred)
    print(autoencoder.m_grad)

    if config["print_results"]:
        current_model = KMP.Model()
        model_part = current_model.CreateModelPart("main_model_part")

        kratos_io.create_out_mdpa(model_part, "GidExampleSwaped")
        kratos_io.print_results_to_gid(model_part, S.T, SP.T)
