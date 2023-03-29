"""Utilities used in various parts of the project."""

import pickle

def read_pkl(filename):
    with open(filename, "rb") as f:
        # Load the dictionaries from the file using the pickle.load() method
        numerical_params = pickle.load(f)
        economic_params = pickle.load(f)
    return numerical_params, economic_params

def to_pkl(dict1, dict2, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict1, f)
        pickle.dump(dict2, f)