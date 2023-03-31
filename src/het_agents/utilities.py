"""Utilities used in various parts of the project."""

import pickle


# TODO: Generalize functions for N dicts
def read_pkl(filename):
    with open(filename, "rb") as f:
        dict1 = pickle.load(f)
        dict2 = pickle.load(f)
    return dict1, dict2


def to_pkl(dict1, dict2, filename):
    with open(filename, "wb") as f:
        pickle.dump(dict1, f)
        pickle.dump(dict2, f)
