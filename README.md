# Location-Aware Maintenance Strategies for Edge Infrastructures

> This repository presents two maintenance strategies, **Lamp** and **Laxus**, designed for updating edge computing infrastructures considering users' locations when performing migration decisions.



## Motivation

Efficient infrastructure maintenance is key to avoiding performance and security issues in computing environments. Despite their contributions, existing maintenance strategies are designed aiming solely at cloud data centers. This repository presents two maintenance strategies (Lamp and Laxus), representing the first steps towards updating edge computing infrastructures while considering previously unexplored requirements such as users' location. While Lamp uses cost functions to determine when and how to update servers properly, Laxus employs a multi-objective genetic algorithm to make Pareto-Optimal maintenance decisions.

## Repository Structure

Within the repository, you'll find the following directories and files, logically grouping common assets used to simulate maintenance of edge computing infrastructures. You'll see something like this:

```
├── datasets/
│   ├── dataset1.json
│   └── example1.json
├── dependencies/
│   └── edge_sim_py-0.1.0-py3-none-any.whl
├── poetry.lock
├── pyproject.toml
├── run_experiments.py
└── simulator/
    ├── algorithms/
    │   ├── best_fit_like.py
    │   ├── first_fit_like.py
    │   ├── greedy_least_batch.py
    │   ├── lamp.py
    │   ├── laxus.py
    │   ├── salus.py
    │   └── worst_fit_like.py
    ├── components_extensions/
    │   ├── edge_server_extensions.py
    │   └── service_extensions.py
    ├── dataset_generator.py
    ├── __main__.py
    └── simulator_extensions.py
```

In the root directory, the `pyproject.toml` file organizes all project dependencies, including the minimum required version of the Python language and the "whl" file containing the simulator core (included in the "dependencies" directory). This file guides the execution of the Poetry library, which installs the simulator securely, avoiding conflicts with external dependencies.

> Modifications made to the pyproject.toml file are automatically inserted into poetry.lock whenever Poetry is called.

The `run_experiments.py` file makes it easy to execute maintenance strategies. For instance, with just a few instructions, we can conduct a complete sensitivity analysis of the maintenance algorithms using different sets of parameters.

The "datasets" directory contains JSON files describing the components that will be simulated during the experiments. We can create custom datasets modifying the `dataset_generator.py` file, located inside the "simulator" directory.

The "algorithms" directory accommodates the source code for the maintenance strategies used in the simulator, while the "components_extensions" directory hosts helper methods that extend the standard functionality of the simulated components. 



## Installation Guide

Project dependencies are available for Linux, Windows and MacOS. However, we highly recommend using a recent version of a Debian-based Linux distribution. The installation below was validated on **Ubuntu 20.04.1 LTS**.

### Prerequisites

We use a Python library called Poetry to manage project dependencies. In addition to selecting and downloading proper versions of project dependencies, Poetry automatically provisions virtual environments for the simulator, avoiding problems with external dependencies. On Linux and MacOS, we can install Poetry with the following command:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

The command above installs Poetry executable inside Poetry’s bin directory. On Unix it is located at `$HOME/.poetry/bin`. We can get more information about Poetry installation at: https://python-poetry.org/docs/#installation.

### Configuration

Considering that we already downloaded the repository, the first thing we need to do is install dependencies using Poetry. To do so, we access the command line in the root directory and type the following command:

```bash
poetry shell
```

The command we just ran creates a virtual Python environment that we will use to run the simulator. Notice that Poetry automatically sends us to the newly created virtual environment. Next, we need to install the project dependencies using the following command:

```bash
poetry install
```

After a few moments, Poetry will have installed all the dependencies needed by the simulator and we will be ready to run the experiments.

## Usage Guide

Our simulator is configured to take arguments via the command line. The most basic arguments, `--dataset` and `--algorithm`, tell the simulator which dataset file (located at `datasets/` directory) and algorithm (located at `simulator/algorithms`) it should execute, respectively. Also, we can pass additional parameters when executing maintenance strategies with configurable hyperparameters (as is the case with Laxus).

### Reproducing Paper Experiments

Below are the commands executed to reproduce the experiments presented in our paper. Please notice that the commands below need to be run inside the virtual environment created by Poetry after the project's dependencies have been successfully installed.

#### Greedy Least Batch

```bash
python3 -B -m simulator --dataset "dataset1" --algorithm "greedy_least_batch"
```

#### Salus

```bash
python3 -B -m simulator --dataset "dataset1" --algorithm "salus"
```

#### Lamp

```bash
python3 -B -m simulator --dataset "dataset1" --algorithm "lamp"
```

#### Laxus

Unlike the other maintenance strategies, Laxus has configurable parameters that modify the behavior of the genetic algorithm it uses to make migration decisions. A description of the custom parameters adopted by Laxus is given below:
- `--n_gen`: determines for how many generations the genetic algorithm will be executed.
- `--pop_size`: determines how many individuals (solutions) will compose the population of the genetic algorithm.
- `--cross_prob`: determines the probability that individuals from the genetic algorithm's population are crossed to generate offsprings.

```bash
python3 -B -m simulator --dataset "dataset1" --algorithm "laxus" --n_gen 800 --pop_size 120 --cross_prob 1
```

## How to Cite

To be defined.

