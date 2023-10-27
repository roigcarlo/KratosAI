import os
import sys
import json

# Disable anoying TF warnings...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import sqlite3 as sql3 
import tensorflow as tf

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
        5 -> "xfoil CL"
        6 -> "kratos CL"

        db_mame: Name of the db.
    '''

    data_map = {}
    db_data = load_simulations(db_mame)

    for row in db_data:
        src_file = row[0]
        src_frwk = row[1]

        # Retrieve the data
        with open(os.path.join("output",f"{src_file}.json"), "r") as src_file_data:
            file_data = json.load(src_file_data)["CL"]
        
            if row[2:] not in data_map:
                datum = [None, None]
                datum[SRC_MAP[src_frwk]] = file_data
                data_map[row[2:]] = datum
            else:
                data_map[row[2:]][SRC_MAP[src_frwk]] = file_data

    # Flatten the data and return
    return np.asarray([[k[0], k[1], k[2], k[3], k[4], v[0], v[1]] for k,v in data_map.items()])


def generate_model():
    # Build a simple network to try to use all the data to preduct Kratos results
    input_size      = 6     # Simulation parameters + Xfoil result
    layer_size      = 18    # Number of Neurons per layer
    hidden_layers   = 4     # Number of Hidden layers

    # Define Some network utilities
    this_actv= tf.keras.activations.relu
    this_init = lambda: tf.keras.initializers.GlorotNormal(seed=None)

    # Define first layer
    model_input = tf.keras.Input(shape=(input_size))
    predictor = tf.keras.layers.Dense(layer_size, use_bias=True, kernel_initializer=this_init())(model_input)

    # Add some fully connected hidden layers of different size (20 as in the paper to begin with)
    for _ in range(hidden_layers):
        predictor = tf.keras.layers.Dense(
            layer_size, 
            activation=this_actv, 
            use_bias=False, 
            kernel_regularizer=tf.keras.regularizers.L2(1e-6),
            # bias_regularizer=tf.keras.regularizers.L2(1e-2),
            # activity_regularizer=tf.keras.regularizers.L2(1e-2),
            kernel_initializer=this_init()
        )(predictor)

    # Add the last layer (prediction of CL)
    predictor = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, use_bias=True, kernel_initializer=this_init())(predictor)

    predictor_model = tf.keras.Model(model_input, predictor)
    predictor_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), run_eagerly=False)

    return predictor_model


if __name__ == "__main__":

    flatten_data = prepare_data("mysqlite3.db")
    predict_model = generate_model()

    # Scale the data to help the training
    scales = []
    for col in range(7):
        max_val = np.max(flatten_data[:,col])
        scales.append(max_val)
        flatten_data[:,col] /= max_val

    # Split the data into training and validation
    split_threshold = 80

    train_data = flatten_data[:split_threshold,:]
    valid_data = flatten_data[split_threshold:,:]

    # Split the data into input and expected outputs
    train_x = train_data[:,:6]
    train_y = train_data[:,6:]

    valid_x = valid_data[:,:6]
    valid_y = valid_data[:,6:]

    # Train the model
    predict_model.fit(
        x=train_x,
        y=train_y,
        batch_size=10,
        shuffle=True,
        validation_data=(valid_x, valid_y),
        epochs=2000
    )

    # Check with validation
    predi_y = predict_model.predict(valid_x)
    for i in range(0, 20):
        orig = valid_x[i][5] * scales[5]
        expc = valid_y[i]    * scales[6]
        pred = predi_y[i]    * scales[6]

        print(f"Pred: {pred[0]:.4f} ({np.abs(pred[0]-expc[0]):.4f}), XFoil: {orig:.4f} ({np.abs(orig-expc[0]):.4f}), Kratos: {expc[0]:.4f}")
