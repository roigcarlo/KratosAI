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

import KratosMultiphysics as KMP
import KratosMultiphysics.gid_output_process as GOP

import KratosMultiphysics.RomApplication as romapp

# from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.RomApplication.randomized_singular_value_decomposition import RandomizedSingularValueDecomposition
# from KratosMultiphysics.RomApplication.structural_mechanics_analysis_rom import StructuralMechanicsAnalysisROM

no_dw = 200
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

def create_out_mdpa(model_part, file_name):
    model_part.AddNodalSolutionStepVariable(KMP.VELOCITY)
    model_part.AddNodalSolutionStepVariable(KMP.PRESSURE)

    model_part.AddNodalSolutionStepVariable(KMP.MESH_VELOCITY)
    model_part.AddNodalSolutionStepVariable(KMP.NEGATIVE_FACE_PRESSURE)

    model_part.AddNodalSolutionStepVariable(KMP.EMBEDDED_VELOCITY)
    model_part.AddNodalSolutionStepVariable(KMP.EXTERNAL_PRESSURE)

    import_flags = KMP.ModelPartIO.READ

    KMP.ModelPartIO(file_name, import_flags).ReadModelPart(model_part)

def print_results_to_gid(model_part, snapshot_matrix, predicted_matrix):

    gid_output = GOP.GiDOutputProcess(
        model_part,
        "PredictDiff",
        KMP.Parameters("""
            {
                "result_file_configuration": {
                    "gidpost_flags": {
                        "GiDPostMode": "GiD_PostAscii",
                        "WriteDeformedMeshFlag": "WriteUndeformed",
                        "WriteConditionsFlag": "WriteConditions",
                        "MultiFileFlag": "SingleFile"
                    },
                    "file_label": "time",
                    "output_control_type": "step",
                    "output_interval": 1.0,
                    "body_output": true,
                    "node_output": false,
                    "skin_output": false,
                    "plane_output": [],
                    "nodal_results": ["PRESSURE", "NEGATIVE_FACE_PRESSURE", "EXTERNAL_PRESSURE", "VELOCITY", "MESH_VELOCITY", "EMBEDDED_VELOCITY"],
                    "nodal_flags_results": ["ISOLATED"],
                    "gauss_point_results": [],
                    "additional_list_files": []
                }
            }
            """
        )
    )

    gid_output.ExecuteInitialize()

    for ts in range(0, snapshot_matrix.shape[0]):
        if not ts%10:
            model_part.ProcessInfo[KMP.TIME] = ts
            gid_output.ExecuteBeforeSolutionLoop()
            gid_output.ExecuteInitializeSolutionStep()

            snapshot = snapshot_matrix[ts]
            predicted = predicted_matrix[ts]

            # print("Snapshot size items:", len(snapshot), "-->", len(snapshot) / 3)

            i = 0
            c = 3
            for node in model_part.Nodes:
                node.SetSolutionStepValue(KMP.VELOCITY_X,0,snapshot[i*c+0])
                node.SetSolutionStepValue(KMP.VELOCITY_Y,0,snapshot[i*c+1])
                node.SetSolutionStepValue(KMP.VELOCITY_Z,0,snapshot[i*c+2])

                node.SetSolutionStepValue(KMP.MESH_VELOCITY_X,0,predicted[i*c+0])
                node.SetSolutionStepValue(KMP.MESH_VELOCITY_Y,0,predicted[i*c+1])
                node.SetSolutionStepValue(KMP.MESH_VELOCITY_Z,0,predicted[i*c+2])

                node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_X,0,abs(snapshot[i*c+0]-predicted[i*c+0]))
                node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_Y,0,abs(snapshot[i*c+1]-predicted[i*c+1]))
                node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_Z,0,abs(snapshot[i*c+2]-predicted[i*c+2]))

                i += 1

            gid_output.PrintOutput()
            gid_output.ExecuteFinalizeSolutionStep()

    gid_output.ExecuteFinalize()


    # ################################## #

    # gid_output.ExecuteInitialize()

    # snapshot = snapshot_matrix[timestep]
    # predicted = predicted_matrix[timestep]

    # print("Snapshot size items:", len(snapshot), "-->", len(snapshot) / 3)

    # i = 0
    # for node in model_part.Nodes:
    #     # print("\tNode:", node.Id)
    #     node.SetSolutionStepValue(KMP.VELOCITY_X,0,snapshot[i*3+0])
    #     node.SetSolutionStepValue(KMP.VELOCITY_Y,0,snapshot[i*3+1])
    #     node.SetSolutionStepValue(KMP.VELOCITY_Z,0,snapshot[i*3+2])
    #     # node.SetSolutionStepValue(KMP.VELOCITY_Z,0,snapshot[i*3+3])

    #     node.SetSolutionStepValue(KMP.MESH_VELOCITY_X,0,predicted[i*3+0])
    #     node.SetSolutionStepValue(KMP.MESH_VELOCITY_Y,0,predicted[i*3+1])
    #     node.SetSolutionStepValue(KMP.MESH_VELOCITY_Z,0,predicted[i*3+2])
    #     # node.SetSolutionStepValue(KMP.MESH_VELOCITY_Z,0,predicted[i*3+3])

    #     node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_X,0,abs(snapshot[i*3+0]-predicted[i*3+0]))
    #     node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_Y,0,abs(snapshot[i*3+1]-predicted[i*3+1]))
    #     node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_Z,0,abs(snapshot[i*3+2]-predicted[i*3+2]))
    #     # node.SetSolutionStepValue(KMP.EMBEDDED_VELOCITY_Z,0,abs(snapshot[i*3+3]-predicted[i*3+3]))

    #     i += 1

    # gid_output.ExecuteBeforeSolutionLoop()
    # gid_output.ExecuteInitializeSolutionStep()
    # gid_output.PrintOutput()
    # gid_output.ExecuteFinalizeSolutionStep()

    # gid_output.ExecuteFinalize()

if __name__ == "__main__":
    ## Run simulation and generate a snapshot dataset
    with open("ProjectParameters.json",'r') as parameter_file:
        parameters = KMP.Parameters(parameter_file.read())
    
    data_inputs = [
        "hdf5_output/result_30.h5",
        "hdf5_output/result_35.h5",
        "hdf5_output/result_40.h5",
        "hdf5_output/result_45.h5",
        "hdf5_output/result_50.h5",
        "hdf5_output/result_55.h5",
        "hdf5_output/result_60.h5",
        # "hdf5_output/result_65.h5",
        # "hdf5_output/result_70.h5",
        # "hdf5_output/result_75.h5",
        # "hdf5_output/result_80.h5",
        # "hdf5_output/result_85.h5",
        # "hdf5_output/result_90.h5",
        # "hdf5_output/result_95.h5",
        # "hdf5_output/result_100.h5",
    ]

    data = build_snapshot_grid(
        data_inputs, 
        [
            # "PRESSURE", 
            "VELOCITY"
        ]
    )

    # data = np.transpose(data)
    begin_matrix_shape = data.shape
    print("K shape:", data.shape)
    # # U,S,V = np.linalg.svd(data, full_matrices=False)
    U,S,_,error = RandomizedSingularValueDecomposition().Calculate(data,1e-6)
    UT = np.transpose(U)

    # SV = np.dot(np.diag(S),np.transpose(V))
    # # USV = np.dot(U, SV)
    t_modes = len(S)
    
    print("U shape:", U.shape)
    print("S shape:", S.shape)
    # print("V shape:", V.shape)
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

    unshuffled_data = data.copy()
    # np.random.shuffle(data)

    ## Create our NN
    decoding_dim = data.shape[1]

    input_data = keras.Input(shape=(decoding_dim,))

    # Tests
    x = layers.Dense(t_modes * 0.8, activation='relu')(input_data)
    x = layers.Dense(t_modes * 0.6, activation='relu')(x)
    x = layers.Dense(t_modes * 0.4, activation='relu')(x)
    x = layers.Dense(t_modes * 0.6, activation='relu')(x)
    x = layers.Dense(t_modes * 0.8, activation='relu')(x)

    decoded = layers.Dense(decoding_dim,  activation='sigmoid')(x)

    autoencoder = keras.Model(input_data, decoded)
    autoencoder.summary()

    # def custom_loss(y_true, y_pred):
    #     y_diff = y_true-y_pred

    #     # y_diff = (y_diff ** 2)

    #     # return tf.keras.backend.mean(y_diff)

    #     return tf.norm(y_diff) / tf.norm(y_true)

    target_error = 1e-6
    def custom_loss(y_true, y_pred):
        y_diff = (y_true-y_pred) # * (data_max - data_min) + data_min
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
    # autoencoder.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.000035, momentum=0.9), loss=custom_loss)
    autoencoder.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.000025, amsgrad=True))
    # autoencoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005), loss='binary_crossentropy')

    ### Training experiment ###
    train_cut = len(data) / len(data_inputs)

    train_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) <  (train_cut * valid)]
    valid_pre = [data[i] for i in range(0, data.shape[0]) if (i % train_cut) >= (train_cut * valid)]

    train_samples = np.array(train_pre)
    valid_samples = np.array(valid_pre)

    train_dataset = np.asarray(train_samples)
    valid_dataset = np.asarray(valid_samples)

    np.random.shuffle(train_dataset)
    np.random.shuffle(valid_dataset)

    # Bigger size and less epochs should return a better model, but we are
    # prototyping now...
    autoencoder.fit(
        train_dataset, train_dataset,
        epochs=10,
        batch_size=1, # int(370*0.7),
        shuffle=True,
        validation_data=(valid_dataset, valid_dataset)
    )

    ### ##################### ###

    # train_subset = train_samples #  * (data_max - data_min) + data_min

    # snapshot_matrix = np.asmatrix(train_subset)
    # snapshot_matrix_norm = np.linalg.norm(snapshot_matrix)

    # pred_subset = autoencoder.predict(train_dataset)

    # prediction_matrix = pred_subset # * (data_max - data_min) + data_min
    # prediction_matrix_norm = np.linalg.norm(prediction_matrix)

    ### ##################### ###

    snapshot_matrix = unshuffled_data
    prediction_matrix = autoencoder.predict(unshuffled_data)

    snapshot_matrix = unshuffled_data * (data_max - data_min) + data_min
    prediction_matrix = prediction_matrix * (data_max - data_min) + data_min
    
    snapshot_matrix = np.dot(UT, snapshot_matrix)
    prediction_matrix = np.dot(UT, prediction_matrix)

    snapshot_matrix_norm = np.linalg.norm(snapshot_matrix)
    prediction_matrix_norm = np.linalg.norm(prediction_matrix)
    
    current_model = KMP.Model()
    model_part = current_model.CreateModelPart("main_model_part")

    print("To print:", 
        "\n\t begin matrix", begin_matrix_shape,
        "\n\t snapshot matrix:", snapshot_matrix.shape,
        "\n\t prediction matrix:", prediction_matrix.shape
    )

    create_out_mdpa(model_part, "GidExampleSwaped")
    print_results_to_gid(model_part, snapshot_matrix, prediction_matrix)

    ### ##################### ###

    # Use the NN to project all the results back to different output
    diff_matrix = abs(snapshot_matrix - prediction_matrix)
    diff_matrix_norm = np.linalg.norm(diff_matrix)

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

    ## End ##
    exit(0)