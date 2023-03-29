"""Tasks for managing the data."""

import pandas as pd
import pytask
import pickle

from heterogeneous_agents.config import BLD, MODELS, SRC
from heterogeneous_agents.data_management import produce_grids
from heterogeneous_agents.utilities import to_pkl


for model in MODELS:


    @pytask.mark.depends_on(
        {
            "scripts": ["produce_grids.py"],
            "data": SRC / "data" / f"parameters_dict_{model}_mkt.pkl",
        },
    )
    @pytask.mark.produces(BLD / "python" / "data" / f"econ_data_{model}_mkt.pkl")
    def task_produce_grids_python(depends_on, produces):
        """Produce all grid, meshes and parameters needed for the model (Python version)."""
        with open("data", "rb") as f:
            # Load the dictionaries from the file using the pickle.load() method
            numerical_params = pickle.load(f)
            economic_params = pickle.load(f)

        updated_economic_params = produce_grids(economic_params, numerical_params)

        to_pkl(updated_economic_params, produces)
