# pylint: disable=invalid-name
"""Contains the basic structure necessary to execute the simulation.
"""
# EdgeSimPy components
from edge_sim_py.simulator import Simulator
from edge_sim_py.component_builders.edge_server_builder import EdgeServerBuilder

# Maintenance-related components
from simulator.simulator_extensions import load_maintenance_attributes
from simulator.simulator_extensions import show_simulated_environment
from simulator.simulator_extensions import update_state
from simulator.simulator_extensions import run
from simulator.simulator_extensions import store_original_state
from simulator.simulator_extensions import restore_original_state
from simulator.simulator_extensions import collect_metrics
from simulator.simulator_extensions import show_results

# Maintenance algorithms
from simulator.algorithms.first_fit_like import first_fit_like
from simulator.algorithms.best_fit_like import best_fit_like
from simulator.algorithms.worst_fit_like import worst_fit_like
from simulator.algorithms.greedy_least_batch import greedy_least_batch
from simulator.algorithms.salus import salus
from simulator.algorithms.laxus import laxus

# Python libraries
import random


VERBOSE = False


def add_maintenance_features(simulator: object, dataset: str):
    # Adding/Overriding EdgeSimPy methods
    Simulator.load_maintenance_attributes = load_maintenance_attributes
    Simulator.update_state = update_state
    Simulator.run = run
    Simulator.store_original_state = store_original_state
    Simulator.restore_original_state = restore_original_state
    Simulator.collect_metrics = collect_metrics
    Simulator.show_results = show_results

    # Calling EdgeSimPy maintenance-related methods
    simulator.load_maintenance_attributes(input_file=dataset)

    if VERBOSE:
        # Displaying simulation info
        show_simulated_environment()


def main():
    """Executes the simulation."""
    # Defining a seed value to allow reproducibility
    random.seed(1)

    # Creating simulator object
    simulator = Simulator()

    # Loading the dataset
    dataset = "datasets/dataset1.json"

    simulator.load_dataset(input_file=dataset)

    # Extending EdgeSimPy with maintenance features
    add_maintenance_features(simulator=simulator, dataset=dataset)

    simulator.run(algorithm=first_fit_like)
    simulator.run(algorithm=worst_fit_like)
    simulator.run(algorithm=best_fit_like)
    simulator.run(algorithm=greedy_least_batch)
    simulator.run(algorithm=salus)
    # simulator.run(algorithm=laxus)

    simulator.show_results(verbosity=VERBOSE)


if __name__ == "__main__":
    main()
