# Solve Heterogeneous Agents

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/JoseLores/het_agents/main.svg)](https://results.pre-commit.ci/latest/github/JoseLores/het_agents/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Usage

To get started, create and activate the environment with

```console
$ conda/mamba env create
$ conda activate heterogeneous_agents
```

To build the project, type

```console
$ pytask
```

## Description

Package to obtain the steady state of heterogeneous agents problems with different
numbers of income states and saving constraints. The model can include linear taxation
to subsidize unemployment (which is the lowest-income state). The package can solve
incomplete market problems by setting interest rate bounds.

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).
It makes a heavy use of:

[Estimagic](https://github.com/OpenSourceEconomics/estimagic).
