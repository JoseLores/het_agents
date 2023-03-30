"""All the general configuration of the project."""
from pathlib import Path

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

MODELS = ["standard", "incomplete"]

ALGORITHMS = [
    "scipy_trust_constr",
    "scipy_lbfgsb",
    "scipy_truncated_newton",
]

__all__ = ["BLD", "SRC", "TEST_DIR", "MODELS", "ALGORITHMS"]
