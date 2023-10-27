import os
import sys
import json

# Disable anoying TF warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import scipy as sp
import sqlite3 as sql3 
import tensorflow as tf
import matplotlib.pyplot as plt

SRC_MAP = {
    "XfoilPanel": 0,
    "KratosFpotentialcompressible": 1
}

def load_simulations(db_name):
    ''' Read the data of the db and returns a flatten_vector where:
        Dumps the sql database into an array
    '''
    conn = sql3.connect(db_name)

    query = f"SELECT * FROM simulations"

    cursor = conn.execute(query)

    return cursor.fetchall()

def prepare_data(db_mame="mysqlite3.db"):
    ''' Read the data of the dumped db and returns a flatten_vector where every row 
        represents a simulation with the following info:

        0 -> "m"
        1 -> "p"
        2 -> "t"
        3 -> "M"
        4 -> "a"
        5 -> "xfoil cp"
        6 -> "kratos cp"

        db_mame: Name of the db.
    '''

    data_map = {}
    db_data = load_simulations(db_mame)

    for row in db_data:
        src_file = row[0]
        src_frwk = row[1]

        # Retrieve the data
        with open(os.path.join("output",f"{src_file}.json"), "r") as src_file_data:
            file_data = json.load(src_file_data)["Cp"]
        
            if row[2:] not in data_map:
                datum = [None, None]
                datum[SRC_MAP[src_frwk]] = file_data
                data_map[row[2:]] = datum
            else:
                data_map[row[2:]][SRC_MAP[src_frwk]] = file_data

    # Flatten the data and return
    return np.asarray([[k[0], k[1], k[2], k[3], k[4], v[0], v[1]] for k,v in data_map.items()], dtype=object)

def interpolate_cp_coordinates(flatten_data):
    ''' Interpolates the values of "Kratos" into to match "xfoil" coordinates.
        This will remove the leading and trailing values of both series

        flatten_data: Parsed data from the DB
    '''

    # From the first simulation in xfoil, get the points
    xfl_cp = np.asarray(flatten_data[0][5])
    kts_cp = np.asarray(flatten_data[0][6])

    # Get the interpolation points
    xfl_x0, xfl_y0 = xfl_cp[:, 0], xfl_cp[:, 1]
    kts_x0, kts_y0 = kts_cp[:, 0], kts_cp[:, 1]

    # Create a filter of valid xfoil and Kratos values (remove the ones that fall outside the interpolation range)
    idx_min_x = np.min(kts_x0)
    idx_max_x = np.max(kts_x0)

    valid_idx = np.where((xfl_x0>idx_min_x) & (xfl_x0<idx_max_x))

    # Make the interpolation
    for row in range(len(flatten_data)):
        print(f"Interpolating row: {row}")

        # Create a 1d interpolator
        xfl_cp = np.asarray(flatten_data[row][5])
        kts_cp = np.asarray(flatten_data[row][6])
        
        # Cut range
        cut_range = 30

        d_idx_x = xfl_cp[120:120+cut_range, 0:2]
        d_idx_k = kts_cp[:, 0:2]

        cp_intrpl1d = sp.interpolate.RBFInterpolator(d_idx_k, kts_cp[:, 2])
        kratos_new_cp = np.column_stack((d_idx_x, cp_intrpl1d(d_idx_x)))

        # Remove the xfoil values outside range
        flatten_data[row][5] = np.asarray(flatten_data[row][5])[120:120+cut_range, :]

        # Assign the Kratos new values
        flatten_data[row][6] = np.asarray(kratos_new_cp)

        # Guarrada borrar (more first)
        # flatten_data[row][5] = np.concatenate((flatten_data[row][5][:30,:], flatten_data[row][5][:,:]))
        # flatten_data[row][6] = np.concatenate((flatten_data[row][6][:30,:], flatten_data[row][6][:,:]))

def generate_model():

    def custom_loss(y_data, y_pred):
        # print(y_data, y_data[0])
        return ((y_data[:,1] - y_pred[:,1])**2)# * ((1 - y_data[:,0] * 0.75))

    # Build a simple network to try to use all the data to preduct Kratos results
    input_size      = 1 # Simulation parameters + Xfoil + coords
    layer_size      = 10     # Number of Neurons per layer
    hidden_layers   = 1     # Number of Hidden layers

    # Define Some network utilities
    this_actv= tf.keras.activations.relu
    this_init = lambda: tf.keras.initializers.HeNormal(seed=None)

    # Define first layer
    model_input = tf.keras.Input(shape=(input_size,8))
    predictor = tf.keras.layers.Dense(layer_size, use_bias=True, kernel_initializer=this_init())(model_input)

    predictor_nl = predictor
    predictor_ln = predictor

    # Add some fully connected hidden layers of different size (20 as in the paper to begin with)
    for _ in range(hidden_layers):
        predictor_nl = tf.keras.layers.Dense(
            layer_size, 
            activation=this_actv, 
            use_bias=True, 
            kernel_regularizer=tf.keras.regularizers.L2(1e-2),
            bias_regularizer=tf.keras.regularizers.L2(1e-4),
            activity_regularizer=tf.keras.regularizers.L2(1e-4),
            kernel_initializer=this_init()
        )(predictor_nl)

    for _ in range(hidden_layers):
        predictor_ln = tf.keras.layers.Dense(
            layer_size, 
            activation=tf.keras.activations.linear, 
            use_bias=True, 
            kernel_regularizer=tf.keras.regularizers.L2(1e-2),
            bias_regularizer=tf.keras.regularizers.L2(1e-4),
            activity_regularizer=tf.keras.regularizers.L2(1e-4),
            kernel_initializer=this_init()
        )(predictor_ln)

    # Add the last layer (prediction of CL)
    # predictor_nl = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer=this_init())(predictor_nl)
    # predictor_ln = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer=this_init())(predictor_ln)

    # predictor = tf.keras.layers.concatenate([predictor_nl, predictor_ln])
    # predictor = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer=this_init(), activation=tf.keras.activations.sigmoid)(predictor)

    # Scale layers
    # predictor = tf.keras.layers.Dense(2, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=this_init())(predictor)
    predictor = predictor_nl
    predictor = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear, use_bias=True, kernel_initializer=this_init())(predictor)

    predictor_model = tf.keras.Model(model_input, predictor)
    predictor_model.compile(loss='mse', optimizer='adam', metrics=['mse'], run_eagerly=False)

    return predictor_model

def plot_control_sim(model, sim_data, minval, maxval):
    # Generate the data for the training process (Simulation parameters + coords + pvalue)
    input_data = []

    for p_idx in range(sim_data[5].shape[0]):
        row_data = [
            [
                sim_data[0], 
                sim_data[1], 
                sim_data[2], 
                sim_data[3],
                sim_data[4],
                (sim_data[5][p_idx,0] - minval[5]) / maxval[5],
                (sim_data[5][p_idx,1] - minval[6]) / maxval[6],
                (sim_data[5][p_idx,2] - minval[7]) / maxval[7]
            ]
        ]

        input_data.append(row_data)

    input_data = np.asarray(input_data)

    XX = sim_data[5][:, 0]
    KX = sim_data[6][:, 0]

    XZ = sim_data[5][:, 2]
    KZ = sim_data[6][:, 2]

    PZ = (model.predict(input_data)[:,-1,-1] * maxval[7]) + minval[7]

    # create a figure and axis object
    fig, ax = plt.subplots()

    # AS HISTOGRAMS
    x_idx = np.argsort(XX)

    XX_sorted = np.array(XX)[x_idx]

    XZ_sorted = np.array(XZ)[x_idx]
    KZ_sorted = np.array(KZ)[x_idx]
    PZ_sorted = np.array(PZ)[x_idx]

    for i in range(len(XZ)):
        print(PZ[i], XZ[i], KZ[i])

    # plot C over X
    ax.plot(XX, KZ, 'o', markersize=4, label="Kratos")
    ax.plot(XX, XZ, 'x', markersize=4, label="XFoil")
    ax.plot(XX, PZ, 'v', markersize=4, label="Predicted")

    ax.legend(loc="upper left")

    print("Chi PREDI", sum((KZ - PZ)**2))
    print("CHI XFOIL", sum((KZ - XZ)**2))

    plt.show()


if __name__ == "__main__":

    flatten_data = prepare_data("mysqlite3.db")
    flatten_data = flatten_data[:2, :]
    
    interpolate_cp_coordinates(flatten_data)

    predict_model = generate_model()

    # Scale the data simulation input data
    minval = []
    maxval = []

    for col in range(5):
        min_val = np.min(flatten_data[:,col])
        max_val = np.max(flatten_data[:,col])
        flatten_data[:,col] /= (max_val)

        minval.append(min_val)
        maxval.append(max_val)

    # For the object type data
    xfl_data = np.asarray([r for r in flatten_data[:,5]])
    kts_data = np.asarray([r for r in flatten_data[:,6]])

    # Scale the values
    for i in range(3):
        min_val_x = np.min(xfl_data[:,:,i])
        min_val_k = np.min(kts_data[:,:,i])

        min_val = np.minimum(min_val_x, min_val_k)
        minval.append(min_val)

        xfl_data[:,:,i] -= min_val
        kts_data[:,:,i] -= min_val

        max_val_x = np.max(xfl_data[:,:,i])
        max_val_k = np.max(kts_data[:,:,i])
        
        max_val = np.maximum(max_val_x, max_val_k)
        maxval.append(max_val)

        xfl_data[:,:,i] /= max_val
        kts_data[:,:,i] /= max_val

    # Generate the data for the training process (Simulation parameters + coords + pvalue)
    data_x = []
    data_y = []

    for s_idx in range(xfl_data.shape[0]):
        for p_idx in range(xfl_data.shape[1]):
            row_x = [
                [flatten_data[s_idx,0], 
                flatten_data[s_idx,1], 
                flatten_data[s_idx,2], 
                flatten_data[s_idx,3],
                flatten_data[s_idx,4],
                xfl_data[s_idx,p_idx,0],
                xfl_data[s_idx,p_idx,1],
                xfl_data[s_idx,p_idx,2]]
            ]

            row_y = [
                kts_data[s_idx,p_idx,2]
            ]

            data_x.append(row_x)
            data_y.append(row_y)

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    
    # # Need to be shuffled before the split
    # np.random.shuffle(input_data)

    # Split the data into training and validation
    split_threshold = int(data_x.shape[0] * 0.99999)

    # Split the data into input and expected outputs
    train_x = data_x[:split_threshold,:]
    valid_x = data_x[split_threshold:,:]

    train_y = data_y[:split_threshold,:]
    valid_y = data_y[split_threshold:,:]

    print(train_x.shape)

    # Train the model
    predict_model.fit(
        x=train_x,
        y=train_y,
        batch_size=train_x.shape[0],
        shuffle=True,
        validation_data=(valid_x, valid_y),
        epochs=1000
    )

    # Check with validation
    predi_y = predict_model.predict(train_x)

    for i in range(0, 1): # valid_x.shape[0]):
        orig = train_x[i][0][-1] * max_val + min_val
        expc = train_y[i][-1] * max_val + min_val
        pred = predi_y[i][-1][0] * max_val + min_val

        print(f"Pred: {pred:.4f} ({np.abs(pred-expc):.4f}), XFoil: {orig:.4f} ({np.abs(orig-expc):.4f}), Kratos: {expc:.4f}")

    plot_control_sim(predict_model, flatten_data[1], minval, maxval)
