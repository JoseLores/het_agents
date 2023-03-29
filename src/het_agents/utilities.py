"""Utilities used in various parts of the project."""

import pickle

import yaml


def read_pkl(filename):
    with open(filename, "rb") as f:
        # Load the dictionaries from the file using the pickle.load() method
        numerical_params = pickle.load(f)
        economic_params = pickle.load(f)
    return numerical_params, economic_params


def to_pkl(dict1, dict2, filename):
    with open(filename, "wb") as f:
        pickle.dump(dict1, f)
        pickle.dump(dict2, f)


def read_yaml(path):
    """Read a YAML file.

    Args:
        path (str or pathlib.Path): Path to file.

    Returns:
        dict: The parsed YAML file.

    """
    with open(path) as stream:
        try:
            out = yaml.safe_load(stream)
        except yaml.YAMLError as error:
            info = (
                "The YAML file could not be loaded. Please check that the path points "
                "to a valid YAML file."
            )
            raise ValueError(info) from error
    return out
