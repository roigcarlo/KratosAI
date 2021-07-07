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

no_dw = 200
no_up = 0

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
                data[variable].append(results_dataset[str(i)]["NodalSolutionStepData"][variable][()])

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

    return data.T

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
        if True:
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

