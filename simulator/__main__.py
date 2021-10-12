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
from simulator.algorithms.lamp import lamp

# Python libraries
import random
import argparse


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


def main(
    dataset: str,
    algorithm: str,
    n_gen: int,
    pop_size: int,
    sampling: str,
    cross: str,
    cross_prob: float,
    mutation: str,
    weights: int,
):
    """Executes the simulation."""
    # Defining a seed value to allow reproducibility
    random.seed(1)

    # Creating simulator object
    simulator = Simulator()

    # Loading the dataset
    dataset = f"datasets/{dataset}.json"

    simulator.load_dataset(input_file=dataset)

    # Extending EdgeSimPy with maintenance features
    add_maintenance_features(simulator=simulator, dataset=dataset)

    arguments = {
        "n_gen": n_gen,
        "pop_size": pop_size,
        "sampling": sampling,
        "cross": cross,
        "cross_prob": cross_prob,
        "mutation": mutation,
        "weights": weights,
    }

    if algorithm in globals():
        simulator.run(algorithm=globals()[algorithm], arguments=arguments)
    else:
        raise Exception("Invalid algorithm name.")

    simulator.show_results(verbosity=VERBOSE)


if __name__ == "__main__":
    # Parsing named arguments from the command line
    parser = argparse.ArgumentParser()

    # Generic arguments
    parser.add_argument("--dataset", "-d", help="Dataset file")
    parser.add_argument("--algorithm", "-a", help="Maintenance algorithm that will be executed")

    # Laxus-specific arguments
    parser.add_argument("--n_gen", help="Number of generations", default="0")
    parser.add_argument("--pop_size", help="Population size", default="0")
    parser.add_argument("--sampling", help="Sampling method", default="int_random")
    parser.add_argument("--cross", help="Crossover method", default="int_ux")
    parser.add_argument("--cross_prob", help="Crossover probability (0.0 to 1.0)", default="0")
    parser.add_argument("--mutation", help="Mutation method", default="int_pm")
    parser.add_argument("--weights", help="Weights for selecting a solution from the Pareto-set", default="0")

    args = parser.parse_args()

    n_gen = int(args.n_gen)
    pop_size = int(args.pop_size)
    sampling = args.sampling
    cross = args.cross
    cross_prob = float(args.cross_prob)
    mutation = args.mutation
    weights = int(args.weights)

    main(
        dataset=args.dataset,
        algorithm=args.algorithm,
        n_gen=n_gen,
        pop_size=pop_size,
        sampling=sampling,
        cross=cross,
        cross_prob=cross_prob,
        mutation=mutation,
        weights=weights,
    )
