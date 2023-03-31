"""Tasks for compiling the paper."""
import shutil

import pytask
from heterogeneous_agents.config import BLD, PAPER_DIR
from pytask_latex import compilation_steps as cs


@pytask.mark.latex(
    script=PAPER_DIR / "heterogeneous_agents.tex",
    document=BLD / "latex" / "heterogeneous_agents.pdf",
    compilation_steps=cs.latexmk(
        options=("--pdf", "--interaction=nonstopmode", "--synctex=1", "--cd"),
    ),
)
@pytask.mark.task(id=document)
def task_compile_document():
    """Compile the document specified in the latex decorator."""


kwargs = {
    "depends_on": BLD / "latex" / "heterogeneous_agents.pdf",
    "produces": BLD.parent.resolve() / "heterogeneous_agents.pdf",
}


@pytask.mark.task(id=document, kwargs=kwargs)
def task_copy_to_root(depends_on, produces):
    """Copy a document to the root directory for easier retrieval."""
    shutil.copy(depends_on, produces)
