"""Tasks for creating the grids that are necessary to solve the models."""


import pytask

from het_agents.config import BLD, MODELS, SRC
from het_agents.data_management import produce_grids
from het_agents.utilities import read_pkl, to_pkl

for model in MODELS:

    @pytask.mark.depends_on(
        {
            "scripts": ["produce_grids.py"],
            "data": SRC / "data" / f"parameters_dict_{model}_mkt.pkl",
        },
    )
    @pytask.mark.produces(BLD / "python" / "data" / f"econ_data_{model}_mkt.pkl")
    def task_produce_grids_python(depends_on, produces):
        """Produce all grid, meshes and parameters needed for the model (Python
        version).
        """
        numerical_params, economic_params = read_pkl(depends_on["data"])

        updated_economic_params = produce_grids(economic_params, numerical_params)

        to_pkl(numerical_params, updated_economic_params, produces)
