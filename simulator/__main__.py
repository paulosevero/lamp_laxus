# pylint: disable=invalid-name
"""Contains the basic structure necessary to execute the simulation.
"""
# EdgeSimPy components
from edge_sim_py.simulator import Simulator

# Maintenance-related components
from simulator.simulator_extensions import load_maintenance_attributes
from simulator.simulator_extensions import show_simulated_environment


def add_maintenance_features(simulator):
    # Adding/Overriding EdgeSimPy methods
    Simulator.load_maintenance_attributes = load_maintenance_attributes

    # Calling EdgeSimPy maintenance-related methods
    simulator.load_maintenance_attributes(input_file="datasets/example1.json")

    # Displaying simulation info
    show_simulated_environment()


def main():
    """Executes the simulation."""
    # Creating simulator object
    simulator = Simulator()

    # Loading the dataset
    simulator.load_dataset(input_file="datasets/example1.json")

    # Extending EdgeSimPy with maintenance features
    add_maintenance_features(simulator)


if __name__ == "__main__":
    main()
