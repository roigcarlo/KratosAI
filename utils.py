import numpy as np

def generate_perturbation(encoded_snapshot):

    pert_sample = np.random.rand(encoded_snapshot.shape[0])
    pert_magnitude = np.linalg.norm(encoded_snapshot)

    return pert_sample * pert_magnitude