from __future__ import print_function, absolute_import, division #makes KratosMultiphysics backward compatible with python 2.6 and 2.7

import json
import math
import h5py
import numpy as np

import keras
import tensorflow as tf
from keras import layers
from itertools import repeat

import matplotlib.pyplot as plt

import KratosMultiphysics
import KratosMultiphysics.RomApplication as romapp

# from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
# from KratosMultiphysics.RomApplication.structural_mechanics_analysis_rom import StructuralMechanicsAnalysisROM

no_dw = 20
no_up = 0
valid = 0.8

def read_snapshot_from_h5py(file_path, variables):
    data = {}

    for variable in variables:
        data[variable] = []

    with h5py.File(file_path, 'r') as f:
        node_indices = f["ModelData"]["Nodes"]["Local"]["Ids"]
        results_dataset = f["ResultsData"]

        number_of_results = len(results_dataset)
        print("Number of results:", number_of_results)
        for i in range(0, number_of_results):

            for variable in variables:
                data[variable].append(results_dataset[str(i)]["NodalSolutionStepData"][variable][()][:3066])

    return data

def read_snapshot_from_h5py_as_tensor(file_path, variables):
    data = []

    with h5py.File(file_path, 'r') as f:
        node_indices = f["ModelData"]["Nodes"]["Local"]["Ids"]
        results_dataset = f["ResultsData"]

        number_of_results = len(results_dataset)
        for i in range(no_dw, number_of_results - no_up):

            nodal_solution_dataset = results_dataset[str(i)]["NodalSolutionStepData"]
            nodal_size = len(node_indices)

            row = np.empty(shape=(nodal_size,0))

            for variable in variables:
                if variable == "PRESSURE":
                    row = np.concatenate((row, np.array(nodal_solution_dataset[variable]).reshape((nodal_size,1))), axis=1)
                if variable == "VELOCITY":
                    row = np.concatenate((row, np.array(nodal_solution_dataset[variable])), axis=1)

            row = np.reshape(row, (row.shape[0] * row.shape[1]))
            data.append(row)
        # print(np.array(data).shape)

    return np.array(data)

def build_snapshot_grid(result_files, variables):
    data = None

    for snapshot in result_files:
        print("Reading results for",snapshot)
        snapshot_data = read_snapshot_from_h5py_as_tensor(snapshot, variables)
        if data is None:
            data = snapshot_data
        else:
            data = np.concatenate((data,snapshot_data), axis=0)
        print("Ok! ",str(len(data)),"results loaded")

    return data

if __name__ == "__main__":
    ## Run simulation and generate a snapshot dataset
    with open("ProjectParameters.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())
    
    data = build_snapshot_grid(
        [
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
        ], [
            "PRESSURE", 
            "VELOCITY"
        ]
    )

    np.random.shuffle(data)

    # data = np.transpose(data)
    print("K shape:", data.shape)
    # # U,S,V = np.linalg.svd(data, full_matrices=False)
    U,S,V,error = RandomizedSingularValueDecomposition().Calculate(data,1e-6)
    UT = np.transpose(U)

    # SV = np.dot(np.diag(S),np.transpose(V))
    # # USV = np.dot(U, SV)
    t_modes = len(S)
    
    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("V shape:", V.shape)
    print("Modes from svd: ", t_modes)

    data = np.dot(U,np.diag(S))
    # Transpose to train colums
    # data = np.transpose(data)

    print("Data shape:", data.shape)

    # exit(0)

    data_min = np.min(data)
    data_max = np.max(data)

    print("RAW - Min:", np.min(data), "Max:", np.max(data))
    
    data = (data - data_min) / (data_max - data_min)

    print("NOR - Min:", np.min(data), "Max:", np.max(data))

    # exit(0)

    # data = read_snapshot_from_h5py("hdf5_output/result.h5", ["PRESSURE", "VELOCITY"])

    ## Create our NN
    encoding_dim = 1500 # int(370 / 1)
    decoding_dim = data.shape[1]

    # print("Modes from SVN: ", t_modes)
    # print("Target autoenc: ", decoding_dim/8)

    input_data = keras.Input(shape=(decoding_dim,))

    # Tests
    x = layers.Dense(t_modes*0.8, activation='relu')(input_data)
    x = layers.Dense(t_modes*0.6, activation='relu')(x)
    x = layers.Dense(t_modes*0.4, activation='relu')(x)
    x = layers.Dense(t_modes*0.6, activation='relu')(x)
    x = layers.Dense(t_modes*0.8, activation='relu')(x)
    # x = layers.Dense(390, activation='relu')(x)
    # x = layers.Dense(300, activation='relu')(x)
    # x = layers.Dense(400, activation='relu')(x)

    decoded = layers.Dense(decoding_dim,  activation='sigmoid')(x)

    # x = layers.Dense(1000, activation='relu')(encoded)
    # x = layers.Dense(5000, activation='relu')(x)

    # decoded = layers.Dense(decoding_dim, activation='sigmoid')(x)

    # Tests
    # x = layers.Conv1D(16, 30, activation='relu', padding="same")(input_data)
    # x = layers.MaxPooling1D(pool_size=2)(x)
    # x = layers.Conv1D(16, 40, activation='relu', padding="same")(x)
    # x = layers.MaxPooling1D(pool_size=2)(x)
    # x = layers.Conv1D(16, 50, activation='relu', padding="same")(x)

    # encoded = layers.Dense(4,  activation='relu')(x)

    # x = layers.Conv1DTranspose(16, 50, activation='relu', padding="same")(encoded)
    # x = layers.UpSampling1D(size=2)(x)
    # x = layers.Conv1DTranspose(16, 40, activation='relu', padding="same")(x)
    # x = layers.UpSampling1D(size=2)(x)
    # x = layers.Conv1DTranspose(16, 30, activation='sigmoid', padding="same")(x)

    # decoded = layers.Conv1DTranspose(1, 3, activation='sigmoid', padding='valid')(x)

    autoencoder = keras.Model(input_data, decoded)
    autoencoder.summary()

    # def custom_loss(y_true, y_pred):
    #     y_diff = y_true-y_pred

    #     # y_diff = (y_diff ** 2)

    #     # return tf.keras.backend.mean(y_diff)

    #     return tf.norm(y_diff) / tf.norm(y_true)

    target_error = 1e-6
    def custom_loss(y_true, y_pred):
        y_diff = y_true-y_pred
        y_diff = y_diff ** 2

        # powers = [1+round((y_diff.shape[1]-1-i/100)/(y_diff.shape[1]-1),2) for i in range(0, y_diff.shape[1])]
        # print(powers)
        # pow_tf = tf.constant(powers)
        
        # y_diff = tf.pow(y_diff,pow_tf)
        # y_diff = tf.math.reduce_sum(y_diff)

        return  y_diff # tf.math.reduce_sum(2**y_diff)
        # return abs(tf.norm(y_true)-tf.norm(y_pred))**10

    ### Compile ###
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # autoencoder.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
    # autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=0.00035)) # Seems to be the best one for "image compression"
    # autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), loss=tf.keras.losses.MeanSquaredError())
    autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.00001))
    # autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss='binary_crossentropy')

    ### Training experiment ###
    train_samples = data[:int(data.shape[0]*valid)]
    valid_samples = data[int(data.shape[0]*valid):]

    # repetitions = 1
    # train_samples = [x for item in train_samples for x in repeat(item, repetitions)]
    # valid_samples = [x for item in valid_samples for x in repeat(item, repetitions)]

    train_dataset = np.asarray(train_samples)
    valid_dataset = np.asarray(valid_samples)

    # Bigger size and less epochs should return a better model, but we are
    # prototyping now...
    autoencoder.fit(
        train_dataset, train_dataset,
        epochs=30,
        batch_size=10, # int(370*0.7),
        shuffle=True,
        validation_data=(valid_dataset, valid_dataset)
    )

    ### ##################### ###

    train_subset = train_samples #  * (data_max - data_min) + data_min

    snapshot_matrix = np.asmatrix(train_subset)
    snapshot_matrix_norm = np.linalg.norm(snapshot_matrix)

    pred_subset = autoencoder.predict(train_dataset)

    prediction_matrix = pred_subset # * (data_max - data_min) + data_min
    prediction_matrix_norm = np.linalg.norm(prediction_matrix)

    # Use the NN to project all the results back to different output

    diff_matrix = abs(snapshot_matrix - prediction_matrix)
    diff_matrix_norm = np.linalg.norm(diff_matrix)

    # print("Autos", autoencoder.layers[1].get_weights())
    # print(prediction_matrix[:2])

    # print(snapshot_matrix, prediction_matrix, diff_matrix)
    # exit()

    print(
        "Snapshot Matrix Norm:",    snapshot_matrix_norm, 
        "Prediction Matrix Norm:",  prediction_matrix_norm, 
        "Diff Matrix Norm",         diff_matrix_norm
    )

    print(
        "Min Error in Matrix:",     np.min(diff_matrix),
        "Max Error in Matrix:",     np.max(diff_matrix)
    )

    print("Norm Error:", diff_matrix_norm / snapshot_matrix_norm)

    fig, axis = plt.subplots(1, 3)

    axis[0].matshow(np.transpose(snapshot_matrix), cmap=plt.get_cmap('jet'))
    axis[1].matshow(np.transpose(prediction_matrix), cmap=plt.get_cmap('jet'))
    axis[2].matshow(np.transpose(diff_matrix) / snapshot_matrix_norm, cmap=plt.get_cmap('jet'))
    plt.show()

    # exit(0)

    # errors = []
    # for i in range(snapshot_matrix.shape[0]):
    #     col_error = snapshot_matrix[i] - prediction_matrix[i] 
    #     errors.append(np.linalg.norm(col_error) / (snapshot_matrix.shape[1]*snapshot_matrix_norm))
    #     print("col_error: ", i, ":", np.linalg.norm(col_error) / (snapshot_matrix.shape[0]*snapshot_matrix_norm))

    # print("Col Errors:", errors)

    ## End ##
    exit(0)