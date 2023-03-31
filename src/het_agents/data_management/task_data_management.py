"""Tasks for creating the grids that are necessary to solve the models."""


import pytask

from het_agents.config import BLD, MODELS, SRC
from het_agents.data_management import produce_grids
from het_agents.utilities import read_pkl, to_pkl

for model in MODELS:
    depends_on = {"data": SRC / "data" / f"parameters_dict_{model}_mkt.pkl"}
    produces = {"results": BLD / "python" / "data" / f"econ_data_{model}_mkt.pkl"}

    @pytask.mark.task(
        id=model,
        kwargs={"model": model, "depends_on": depends_on, "produces": produces},
    )
    def task_produce_grids_python(depends_on, model, produces):
        """Produce all grid, meshes and parameters needed for the model (Python
        version).
        """
        numerical_params, economic_params = read_pkl(depends_on["data"])
        updated_economic_params = produce_grids(economic_params, numerical_params)
        to_pkl(numerical_params, updated_economic_params, produces["results"])
