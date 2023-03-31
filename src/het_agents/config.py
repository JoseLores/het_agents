"""All the general configuration of the project."""
from pathlib import Path

SRC = Path(__file__).parent.resolve()
BLD = SRC.joinpath("..", "..", "bld").resolve()

TEST_DIR = SRC.joinpath("..", "..", "tests").resolve()
PAPER_DIR = SRC.joinpath("..", "..", "paper").resolve()

MODELS = ["standard", "incomplete"]

ALGORITHMS = [
    "scipy_truncated_newton",
    "scipy_lbfgsb",
    "scipy_powell",
]

__all__ = ["BLD", "SRC", "TEST_DIR", "MODELS", "ALGORITHMS"]
