import os
import sys
import csv
import json

# Disable anoying TF warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import scipy as sp
import sqlite3 as sql3 
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import backend as K

LF_DATA = 0
HF_DATA = 1

SRC_MAP = {
    "XfoilPanel"                        : LF_DATA,
    "KratosFpotentialcompressible"      : HF_DATA,
    "KratosFpotentialtransonic"         : HF_DATA,
    "CODAEuler"                         : HF_DATA
}

LF_FRWK = "KratosFpotentialtransonic"
HF_FRWK = "CODAEuler"

def load_simulations_from_db(db_name):
    ''' Read the data of the db and returns a flatten_vector where:
        Dumps the sql database into an array
    '''
    conn = sql3.connect(db_name)

    query = f"SELECT * FROM simulations"

    cursor = conn.execute(query)

    return cursor.fetchall()

def prepare_data_from_db(db_mame="mysqlite3.db"):
    ''' Read the data of the dumped db and returns a flatten_vector where every row 
        represents a simulation with the following info:

        0 -> "m"
        1 -> "p"
        2 -> "t"
        3 -> "M"
        4 -> "a"
        5 -> "LFDATA CP"
        6 -> "HFDATA CP"

        db_mame: Name of the db.
    '''

    data_map = {}
    db_data = load_simulations_from_db(os.path.join(db_mame,"mysqlite3.db"))

    for row in db_data:
        src_file = row[0]
        src_frwk = row[1]
        data_target = None
        # Retrieve the data
        if src_frwk == LF_FRWK:
            data_target = 0
        if src_frwk == HF_FRWK:
            data_target = 1
        
        if data_target is not None:
            with open(os.path.join(db_mame,"output",f"{src_file}.json"), "r") as src_file_data:
                file_data = json.load(src_file_data)["Cp"]
            
                if row[2:] not in data_map:
                    datum = [None, None]
                    datum[data_target] = file_data
                    data_map[row[2:]] = datum
                else:
                    data_map[row[2:]][data_target] = file_data

    # Flatten the data and return
    return np.asarray([[k[0], k[1], k[2], k[3], k[4], v[0], v[1]] for k,v in data_map.items()], dtype=object)

def prepare_data_raw(raw_path):
    ''' Read the data of the dumped db and returns a flatten_vector where every row 
        represents a simulation with the following info:

        0  -> "R_le"
        1  -> "L_le"
        2  -> "rr_le"
        3  -> "psi1_ledeg"
        4  -> "psi2_ledeg"
        5  -> "L1_te"
        6  -> "L2_te"
        7  -> "theta1_tedeg"
        8  -> "theta2_tedeg"
        9  -> "M"
        10 -> "a"
        11 -> "xfoil cp"
        12 -> "kratos cp"

        db_mame: Name of the db.
    '''

    print(f"Directory {raw_path} has {len(os.listdir(raw_path))} files")

    data_map = {}

    for src_file_data_name in os.listdir(raw_path):
        with open(os.path.join(raw_path,f"{src_file_data_name}"), "r") as src_file_data:
            as_json = json.load(src_file_data)

            simulation_setting = as_json["Settings"]
            simulation_model   = as_json["Model"]

            file_data = as_json["Cp"]

            control = np.max(np.abs(np.asarray(file_data)[:,2]))

            if control >= 10:
                print(f"Excluding: {src_file_data_name} -- {control}...")

            key = tuple(simulation_setting.values())

            if key not in data_map:
                data_map[key] = [None, None]
            if control < 10:
                data_map[key][SRC_MAP[simulation_model]] = file_data

    inva_map = [k for k,v in data_map.items() if v[0] == None or  v[1] == None]

    print(f"The following invalid entries were found: {len(inva_map)}")

    data_map = {k:v for k,v in data_map.items() if v[0] != None and v[1] != None}

    # Flatten the data and return
    return np.asarray([[k[-11], k[-10], k[-9], k[-8], k[-7], k[-6], k[-5], k[-4], k[-3], k[-2], k[-1], v[0], v[1]] for k,v in data_map.items()], dtype=object)

def remove_duplicates(data):
    ''' Tries to remove duplicated entried from the db if present
    '''

    np.sort(data, axis=0)
    unique_rows, indices, inverse_indices = np.unique(data, axis=0, return_index=True, return_inverse=True)

    # print(f"Found {len(unique_rows)} unique points out of {data.shape[0]}")

    return unique_rows

def has_shock(row):
    ''' Returns true if a shock is detected in the profile or false if not. 

        @slope_fract: Minimum value of the slope in any given point to be considered a shock
    ''' 

    slope_fract = 1

    try:
        top = row[row[:,1] > 0]
        sort_ord = np.argsort(top[:,0])
        slop_ord = np.abs(np.ediff1d(top[sort_ord][:,2], to_end=None, to_begin=None))
        return np.count_nonzero(slop_ord > slope_fract)
    except:
        return 0

def filter_outlayer_results(flatten_data, chi=2):
    ''' Filter results with chi crtieria.
    '''
    
    print(f"Excluding {len([row for row in flatten_data if sum((np.asarray(row[-1])[:,2] - np.asarray(row[-2])[:,2])**2) > chi])} lf results with high (>{chi}) chi error...")
    
    return np.asarray([row for row in flatten_data if sum((np.asarray(row[-1])[:,2] - np.asarray(row[-2])[:,2])**2) < chi])

def filter_non_shock(flatten_data):
    ''' Filter results with shock crtieria.
    '''

    return [row for row in flatten_data if has_shock(np.asarray(row[-1])) > 0]

def interpolate_cp_coordinates(flatten_data):
    ''' Interpolates the values from both LF and HF data so the training sizes match.

        TODO: Possible delegate this to the network by fine tunning the first layer with variable size or
              find a way to encode different sizes in the input
        This will remove the leading and trailing values of both series

        flatten_data: Parsed data from the DB
    '''

    # Get the points from both simulations
    lfd_cp = np.asarray(flatten_data[0][-2])
    hfd_cp = np.asarray(flatten_data[0][-1])

    # Get the interpolation points
    lfd_x0, lfd_y0 = lfd_cp[:, 0], lfd_cp[:, 1]
    hfd_x0, hfd_y0 = hfd_cp[:, 0], hfd_cp[:, 1]

    # Create a filter of valid LF and HF values (remove the ones that fall outside the interpolation range)
    idx_min_x = np.min(hfd_x0)
    idx_max_x = np.max(hfd_x0)

    valid_idx = np.where((lfd_x0>idx_min_x) & (lfd_x0<idx_max_x))

    # Make the interpolation
    for row in range(len(flatten_data)):
        # Split the low fidelity and high fidelity data
        lfd_cp = np.asarray(flatten_data[row][-2])
        hfd_cp = np.asarray(flatten_data[row][-1])
        
        # Remove duplicates from the data
        lfd_unique = remove_duplicates(lfd_cp)
        hfd_unique = remove_duplicates(hfd_cp)

        # Select the number of points
        n_points = 400 # min(lfd_cp.shape[0], hfd_cp.shape[0])
        d_points = hfd_unique[0:n_points, 0:2]
        d_points = d_points[:,:][d_points[:,1] > 0 ]
        d_points = d_points[:200]

        # Make an interpolator using the HF data
        try:
            d_idx_lf = lfd_unique[:,0:2][lfd_unique[:,1] > 0]
            d_idx_hf = hfd_unique[:,0:2][hfd_unique[:,1] > 0]
        except Exception as e:
            print("Failed inserting", flatten_data[row])
        
        cp_intrpl1d_lf = sp.interpolate.RBFInterpolator(d_idx_lf, lfd_unique[:, 2][lfd_unique[:,1] > 0])
        cp_intrpl1d_hf = sp.interpolate.RBFInterpolator(d_idx_hf, hfd_unique[:, 2][hfd_unique[:,1] > 0])

        # Interpolate both lf and hf data
        lf_new_cp = np.column_stack((d_points, cp_intrpl1d_lf(d_points)))
        hf_new_cp = np.column_stack((d_points, cp_intrpl1d_hf(d_points)))

        # Assign the Interpolated values
        flatten_data[row][-2] = np.asarray(lf_new_cp) # np.asarray(flatten_data[row][-2])[0:n_points, :]
        flatten_data[row][-1] = np.asarray(hf_new_cp)

        # # Assign the Interpolated values
        # flatten_data[row][-2] = flatten_data[row][-2][::4]
        # flatten_data[row][-1] = flatten_data[row][-1][::4]

        if row % 10 == 0: 
            print(f"Interpolating row: {row}:{flatten_data[row][-2].shape}")

def generate_model():
    ''' Creates a NN model to train 
    '''

    # def custom_loss(y_data, y_pred):
    #     return tf.abs(y_data - y_pred)/y_data

    def custom_loss(y_data, y_pred):
        return (y_data - y_pred)**2

    # Build a simple network to try to use all the data to predict the hf results
    input_size      = 1     # Simulation parameters + LF + coords
    input_points    = 200   # Number of sampling points
    model_features  = 3     # Number of features (sim_params, x, y, aprox, etc...)
    macha_features  = 2     # Number of features (sim_params, x, y, aprox, etc...)
    layer_size      = 40    # Number of Neurons per layer
    hidden_layers   = 4     # Number of Hidden layers

    # Define Some network utilities
    this_actv= tf.keras.activations.relu
    this_init = lambda: tf.keras.initializers.HeNormal(seed=None)

    # Define first layer
    model_input = tf.keras.Input(shape=(input_points, model_features))      # Coordinates and CP input
    macha_input = tf.keras.Input(shape=(macha_features))                    # Mach and angle input

    predictor = tf.keras.layers.Dense(layer_size, use_bias=True, kernel_initializer=this_init())(model_input)

    predictor_cv = model_input
    predictor_cv = tf.keras.layers.Conv1D(8, 3, strides=1, padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), data_format='channels_last', input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.MaxPooling1D(2, strides=2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1D(16, 3, strides=1, padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), data_format='channels_last', input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.MaxPooling1D(2, strides=2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1D(32, 3, strides=1, padding='same',  activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), data_format='channels_last', input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.MaxPooling1D(2, strides=2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1D(64, 3, strides=1, padding='same',  activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), data_format='channels_last', input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.MaxPooling1D(2, strides=2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1D(128, 3, strides=1, padding='same',  activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), data_format='channels_last', input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.MaxPooling1D(2, strides=2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1D(256, 3, strides=1, padding='same',  activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), data_format='channels_last', input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.MaxPooling1D(2, strides=2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1DTranspose(128, 3, strides=1, padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.UpSampling1D(2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1DTranspose(64, 3, strides=1, padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.UpSampling1D(2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1DTranspose(32, 3, strides=1, padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.UpSampling1D(2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1DTranspose(16, 3, strides=1, padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.UpSampling1D(2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1DTranspose(8, 3, strides=1, padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.UpSampling1D(2)(predictor_cv)

    predictor_cv = tf.keras.layers.Conv1DTranspose(1, 3, strides=1, padding='same', activation='relu', bias_regularizer=tf.keras.regularizers.L1L2(1e-2), kernel_regularizer=tf.keras.regularizers.L1L2(1e-2), input_shape=(input_points, model_features))(predictor_cv)
    predictor_cv = tf.keras.layers.UpSampling1D(2)(predictor_cv)

    # predictor = predictor_cv

    # predictor_cv = tf.keras.layers.Flatten()(predictor_cv)
    # predictor_cv = tf.keras.layers.Dense(input_points * layer_size, use_bias=True, kernel_initializer=this_init())(predictor_cv)
    
    # # Concatenate the macha input with the model input
    # ln_nl_layer_input = tf.keras.layers.Flatten()(predictor_cv)
    ln_nl_layer_input = tf.keras.layers.Reshape((192,), input_shape=(192,1))(predictor_cv)
    ln_nl_layer_input = tf.keras.layers.concatenate([ln_nl_layer_input, macha_input], axis=1)
    ln_nl_layer_input = tf.keras.layers.Dense(input_points, use_bias=True, activation=this_actv)(ln_nl_layer_input)
    ln_nl_layer_input = tf.keras.layers.Dense(input_points, use_bias=True, activation=this_actv)(ln_nl_layer_input)
    ln_nl_layer_input = tf.keras.layers.Dense(input_points, use_bias=True, activation=this_actv)(ln_nl_layer_input)
    predictor = tf.keras.layers.Reshape((200,1), input_shape=(200,))(ln_nl_layer_input)

    # # Add some fully connected hidden layers of different size (20 as in the paper to begin with)
    # predictor_nl = [ln_nl_layer_input]
    # predictor_nl_ls = 100
    # for _ in range(hidden_layers):
    #     predictor_nl.append(tf.keras.layers.Dense(
    #         predictor_nl_ls, 
    #         activation=this_actv, 
    #         use_bias=True, 
    #         kernel_regularizer=tf.keras.regularizers.L2(1e-7),
    #         bias_regularizer=tf.keras.regularizers.L2(1e-7),
    #         activity_regularizer=tf.keras.regularizers.L2(1e-7),
    #         kernel_initializer=this_init()
    #     )(predictor_nl[-1]))
    #     predictor_nl_ls /= 2

    # predictor_ln = [ln_nl_layer_input]
    # predictor_ln_ls = 100
    # for _ in range(hidden_layers):
    #     predictor_ln.append(tf.keras.layers.Dense(
    #         predictor_ln_ls, 
    #         use_bias=True, 
    #         kernel_regularizer=tf.keras.regularizers.L2(1e-7),
    #         bias_regularizer=tf.keras.regularizers.L2(1e-7),
    #         activity_regularizer=tf.keras.regularizers.L2(1e-7),
    #         kernel_initializer=this_init()
    #     )(predictor_ln[-1]))
    #     predictor_ln_ls /= 2

    # predictor = tf.keras.layers.concatenate([predictor_ln[-1],  predictor_nl[-1], predictor_cv])
    # predictor = tf.keras.layers.Dense(24, use_bias=True, kernel_regularizer=tf.keras.regularizers.L2(1e-7), bias_regularizer=tf.keras.regularizers.L2(1e-7), kernel_initializer=this_init(), activation=this_actv)(predictor)
    # predictor = tf.keras.layers.Dense(48, use_bias=True, kernel_regularizer=tf.keras.regularizers.L2(1e-7), bias_regularizer=tf.keras.regularizers.L2(1e-7), kernel_initializer=this_init(), activation=this_actv)(predictor)
    # predictor = tf.keras.layers.Dense(96, use_bias=True, kernel_regularizer=tf.keras.regularizers.L2(1e-7), bias_regularizer=tf.keras.regularizers.L2(1e-7), kernel_initializer=this_init(), activation=this_actv)(predictor)
    # predictor = tf.keras.layers.Dense(input_points, use_bias=True, kernel_regularizer=tf.keras.regularizers.L2(1e-7), bias_regularizer=tf.keras.regularizers.L2(1e-7), kernel_initializer=this_init())(predictor)

    predictor_model = tf.keras.Model([model_input, macha_input], predictor, name="HF_Predictor")
    predictor_model.compile(loss=custom_loss, optimizer=tf.keras.optimizers.Adam(lr=0.02), metrics=['mse'], run_eagerly=False)

    predictor_model.summary()

    return predictor_model

def plot_control_sim(model, flatten_data, minval, maxval, split_threshold, dataset_name):
    # Generate the data for the training process (Simulation parameters + coords + pvalue)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # chi LF, chi PRED, is_validation
    results_data = []

    # x, y_pred, y_LF, x_HF
    results_coords = []

    for i in range(len(flatten_data[:])):
        sim_data = flatten_data[i]

        row_x = []
        for p_idx in range(lf_data.shape[1]):
            row_x.append([
                # sim_data[-7],
                # sim_data[-6],
                # sim_data[-5],
                # sim_data[-4],
                # sim_data[-3],
                (sim_data[-2][p_idx,0] - minval[-3]) / maxval[-3],
                (sim_data[-2][p_idx,1] - minval[-2]) / maxval[-2],
                (sim_data[-2][p_idx,2] - minval[-1]) / maxval[-1]
            ])

        # As np array
        input_data = np.asarray([row_x])
        input_g = np.asarray([(sim_data[-4], sim_data[-3])])

        LF_X = sim_data[-2][:, 0]
        HF_X = sim_data[-1][:, 0]

        LF_Z = -sim_data[-2][:, 2]
        HF_Z = -sim_data[-1][:, 2]

        x_idx = np.argsort(LF_X)

        print(f"{input_data.shape=}")

        PZ = model.predict((input_data,input_g))

        print(f"{PZ[0].shape=}")

        PZ = -((PZ[0,:,0] * maxval[-1]) + minval[-1])

        # plot C over X
        if sum((HF_Z - LF_Z)**2) > 10:
            ax.plot(LF_X, HF_Z, 'o', markersize=4, label=f"HFDAT_{i}")
            ax.plot(LF_X, LF_Z, 'x', markersize=4, label=f"LFDAT_{i}")
            ax.plot(LF_X, PZ, 'v', markersize=4, label=f"PREDI_{i}")

        print(f"Chi PREDI({i}): {sum((HF_Z - PZ)**2)}")
        print(f"CHI LFDAT({i}): {sum((HF_Z - LF_Z)**2)}")

        results_data.append([sum((HF_Z - PZ)**2), sum((HF_Z - LF_Z)**2), i > split_threshold])
        results_coords.append([LF_X.tolist(), PZ.tolist(), LF_Z.tolist(), HF_Z.tolist()])

    ax.legend(loc="upper left")

    with open(os.path.join("results",dataset_name.replace("/","_") + "_chi_" + str(os.getpid()) + ".csv"), 'w') as results_file:
        wr = csv.writer(results_file)
        wr.writerows(results_data)

    with open(os.path.join("results",dataset_name.replace("/","_") + "_xyz_" + str(os.getpid()) + ".csv"), 'w') as results_file:
        json.dump({"meta":[sim_data[-4], sim_data[-3]], "data":results_coords}, results_file)

    # plt.show()

def plot_result_space(flatten_data, train_fract):

    to_ = int(flatten_data.shape[0] * (1-train_fract))

    # a,M axis
    a_values = np.array([t[3] for t in flatten_data])
    b_values = np.array([t[4] for t in flatten_data])

    # Markers 
    flatten_data = np.array(flatten_data)
    s_values = np.array([has_shock(np.asarray(row[-1])) for row in flatten_data])

    # Plot train data
    train_a = a_values[:-to_]
    train_b = b_values[:-to_]
    train_s = s_values[:-to_]

    plt.scatter(train_a[train_s == 0], train_b[train_s == 0], color='blue', label='Training', marker='x')
    plt.scatter(train_a[train_s != 0], train_b[train_s != 0], color='cyan', label='Training', marker='o')

    # Plot valid data
    valid_a = a_values[-to_:]
    valid_b = b_values[-to_:]
    valid_s = s_values[-to_:]

    plt.scatter(valid_a[valid_s == 0], valid_b[valid_s == 0], color='red', label='Training', marker='x')
    plt.scatter(valid_a[valid_s != 0], valid_b[valid_s != 0], color='orange', label='Training', marker='o')

    plt.xlabel('M')
    plt.ylabel('a')
    plt.title('Scatterplot of M vs. a')
    plt.show()

    exit()

if __name__ == "__main__":

    # Generate the model and leave it there
    predict_model = generate_model()

    # Read the Db
    dataset_names = [
        "dataset_02_10_2023_0",
        "dataset_02_10_2023_1",
        # "dataset_05_10_2023_0",
        # "dataset_05_10_2023_1",
    ]

    flatten_data = np.empty([0, 7])
    for dataset_name in dataset_names:
        db_path = os.path.join("datasets",dataset_name)
        flatten_data = np.vstack([flatten_data,prepare_data_from_db(db_path)])
        print(flatten_data.shape)

    # Shuffle the simulations. This is mandatory while loading multiple db
    np.random.shuffle(flatten_data)

    # Define the training data fraction
    train_fract = 0.80

    # Plot the loaded results()
    # plot_result_space(flatten_data, train_fract)

    # Select only N simulations from the DB ( usefull for prototyping )
    flatten_data = flatten_data[:, :]

    # Filter simulations without shock (or with a very small error)
    flatten_data = filter_non_shock(flatten_data)
    print(len(flatten_data))

    # Prepare the results for processing
    interpolate_cp_coordinates(flatten_data)

    # Filter results with abnormal values
    flatten_data = filter_outlayer_results(flatten_data,2000)

    # Scale the data simulation input data
    minval = []
    maxval = []

    num_features = len(flatten_data[0]) - 2

    for col in range(num_features):
        min_val = np.min(flatten_data[:,col])
        max_val = np.max(flatten_data[:,col])

        flatten_data[:,col] /= (max_val)

        minval.append(min_val)
        maxval.append(max_val)

    # For the object type data
    lf_data = np.asarray([r for r in flatten_data[:,-2]])
    hf_data = np.asarray([r for r in flatten_data[:,-1]])

    # Scale the values
    for i in range(3):
        min_val_x = np.min(lf_data[:,:,i])
        min_val_k = np.min(hf_data[:,:,i])

        min_val = np.minimum(min_val_x, min_val_k)
        minval.append(min_val)

        lf_data[:,:,i] -= min_val
        hf_data[:,:,i] -= min_val

        max_val_lf = np.max(lf_data[:,:,i])
        max_val_hf = np.max(hf_data[:,:,i])
        
        max_val = np.maximum(max_val_lf, max_val_hf)
        maxval.append(max_val)

        lf_data[:,:,i] /= max_val
        hf_data[:,:,i] /= max_val

    # Generate the data for the training process (Simulation parameters + coords + pvalue)
    data_x = []
    data_y = []
    data_g = []

    print(f"{flatten_data.shape=}")

    for s_idx in range(flatten_data.shape[0]):
        row_x = []
        row_y = []

        for p_idx in range(lf_data.shape[1]):
            row_x.append([
                # flatten_data[s_idx,-7],
                # flatten_data[s_idx,-6],
                # flatten_data[s_idx,-5],
                # flatten_data[s_idx,-4],
                # flatten_data[s_idx,-3],
                lf_data[s_idx,p_idx,0],
                lf_data[s_idx,p_idx,1],
                lf_data[s_idx,p_idx,2]
            ])

            row_y.append(
                hf_data[s_idx,p_idx,2]
            )

        data_x.append(row_x)
        data_y.append(row_y)
        data_g.append([flatten_data[s_idx,-4], flatten_data[s_idx,-3]])

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    data_g = np.asarray(data_g)

    print(f"{data_x.shape=}, {data_y.shape=}")

    # Split the data into training and validation
    split_threshold = int(data_x.shape[0] * train_fract)

    # Split the data into input and expected outputs
    train_x = data_x[:split_threshold,:]
    valid_x = data_x[split_threshold:,:]

    train_y = data_y[:split_threshold,:]
    valid_y = data_y[split_threshold:,:]

    train_g = data_g[:split_threshold,:]
    valid_g = data_g[split_threshold:,:]

    # Set num epochs
    num_epochs = 4000

    # Train the model
    predict_model.fit(
        x=(train_x, train_g),
        y=train_y,
        batch_size=int(train_x.shape[0]/4),
        shuffle=True,
        validation_data=((valid_x, valid_g), valid_y),
        epochs=num_epochs
    )

    # Save the model
    predict_model.save(f"models/train_{num_epochs}.h5")

    # Check with validation
    predi_y = predict_model.predict((train_x,train_g))

    print(f"{predi_y.shape=}")

    for i in range(0, 30): # valid_x.shape[0]):
        orig = train_x[0][i][-1] * max_val + min_val
        expc = train_y[0][i]     * max_val + min_val
        pred = predi_y[0][i][-1] * max_val + min_val

        print(f"PREDI: {pred:.4f} ({np.abs(pred-expc):.4f}), LFDAT: {orig:.4f} ({np.abs(orig-expc):.4f}), HFDAT: {expc:.4f}")

    plot_control_sim(predict_model, flatten_data, minval, maxval, split_threshold, dataset_name)
