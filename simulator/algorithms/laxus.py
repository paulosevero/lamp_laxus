""" Contains Laxus maintenance algorithm."""
# EdgeSimPy components
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.service import Service

# Python libraries
import numpy as np
import networkx as nx
from multiprocessing.pool import ThreadPool

# Pymoo components
from pymoo.util.display import Display
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation


VERBOSE = False
PARALLEL = False
N_THREADS = 4
WEIGHT_OPTIONS = [
    [1, 1, 1],
    [1, (1 / 2), (1 / 3)],
    [1, (1 / 3), (1 / 2)],
    [(1 / 2), 1, (1 / 3)],
    [(1 / 3), 1, (1 / 2)],
    [(1 / 2), (1 / 3), 1],
    [(1 / 3), (1 / 2), 1],
]


class MyDisplay(Display):
    """Creates a visualization on how the genetic algorithm is evolving throughout the generations."""

    def _do(self, problem: object, evaluator: object, algorithm: object):
        """Defines the way information about the genetic algorithm is printed after each generation.

        Args:
            problem (object): Instance of the problem being solved.
            evaluator (object): Object that makes modifications before calling the problem's evaluate function.
            algorithm (object): Algorithm being executed.
        """
        super()._do(problem, evaluator, algorithm)

        outdated_servers_occupied = int(np.min(algorithm.pop.get("F")[:, 0]))
        useless_migrations = int(np.min(algorithm.pop.get("F")[:, 1]))
        sla_violations = int(np.min(algorithm.pop.get("F")[:, 2]))

        self.output.append("Outdated SVs", outdated_servers_occupied)
        self.output.append("Migr. Dur.", useless_migrations)
        self.output.append("SLA viol.", sla_violations)


class PlacementProblem(Problem):
    """Describes the application placement as an optimization problem."""

    def __init__(self, **kwargs):
        """Initializes the problem instance."""
        self.services_on_outdated_hosts = len([1 for service in Service.all() if not service.server.updated])
        super().__init__(
            n_var=self.services_on_outdated_hosts,
            n_obj=3,
            n_constr=1,
            xl=1,
            xu=EdgeServer.count(),
            type_var=int,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluates solutions according to the problem objectives.

        Args:
            x (list): Solution or set of solutions that solve the problem.
            out (dict): Output of the evaluation function.
        """
        if PARALLEL:
            thread_pool = ThreadPool(N_THREADS)
            output = thread_pool.map(self.get_fitness_score_and_constraints, x)
            thread_pool.close()
        else:
            output = [self.get_fitness_score_and_constraints(solution=solution) for solution in x]

        out["F"] = np.array([item[0] for item in output])
        out["G"] = np.array([item[1] for item in output])

    def get_fitness_score_and_constraints(self, solution: list) -> tuple:
        """Calculates the fitness score and penalties of a solution based on the problem definition.

        Args:
            solution (list): Solution that solves the problem.

        Returns:
            tuple: Output of the evaluation function containing the fitness scores of the solution and its penalties.
        """
        topology = Topology.first()

        # Declaring objectives
        sla_violations = 0
        migrations_duration = []
        undesired_migrations = 0
        vulnerable_servers = 0

        # Gathering edge servers capacity
        edge_servers_free_capacity = [edge_server.capacity - edge_server.demand for edge_server in EdgeServer.all()]

        # Gathering the list of services hosted by outdated hosts (these servers will possibly be relocated)
        services_hosted_by_outdated_servers = [service for service in Service.all() if not service.server.updated]

        # Objective 1: minimize the number of outdated servers
        for server in EdgeServer.outdated():
            if server.id in solution:
                vulnerable_servers += 1

        # Applying the placement scheme suggested by the solution
        for service_id, server_id in enumerate(solution, 1):
            server = EdgeServer.instances[server_id - 1]
            service = services_hosted_by_outdated_servers[service_id - 1]
            user = service.application.users[0]

            # Checking if the current service is migrated by the solution
            if service.server != server:
                # Updating the demand of edge servers involved in the migration
                edge_servers_free_capacity[service.server.id - 1] += service.demand
                edge_servers_free_capacity[server.id - 1] -= service.demand

                # Objective 2 (part 1): minimize the average migration duration
                migration_path = nx.shortest_path(
                    G=topology,
                    source=service.server.base_station,
                    target=server.base_station,
                    weight="bandwidth",
                )
                migration_duration = service.get_migration_time(path=migration_path)
                migrations_duration.append(migration_duration)

                # Objective 2 (part 2): minimize the number of undesired migrations. Here, undesired migrations
                # are those that don't lead an outdated server drained or target outdated servers
                if service.server.id in solution or not server.updated:
                    undesired_migrations += 1

            # Objective 3: minimize the number of SLA violations
            sla = user.delay_slas[service.application]
            shortest_path = nx.shortest_path(
                G=topology,
                source=user.base_station,
                target=server.base_station,
                weight=lambda u, v, d: topology[u][v]["delay"]
                if topology[u][v]["demand"] + service.application.network_demand <= topology[u][v]["bandwidth"]
                else topology[u][v]["delay"] * 10,
                method="dijkstra",
            )
            delay = user.base_station.wireless_delay + topology.calculate_path_delay(shortest_path)
            if delay > sla:
                sla_violations += 1

        # Aggregating the different fitness scores of the evaluated solution
        if len(migrations_duration) == 0:
            migration_score = 0
        else:
            migration_score = (sum(migrations_duration) / len(migrations_duration)) * (1 + undesired_migrations)

        fitness = (
            vulnerable_servers,
            migration_score,
            sla_violations,
        )

        # Calculating the number of overloaded servers (which is the problem constraint and represents a penalty)
        overloaded_servers = sum([1 for item in edge_servers_free_capacity if item < 0])

        return (fitness, overloaded_servers)


def min_max_norm(x, minimum, maximum) -> float:
    """Rescales a variable to a range between 0 and 1 using the rescaling method (also known as min-max normalization).

    Args:
        x (number): Variable that will be rescaled.
        minimum (number): Minimum value from the dataset.
        maximum (number): Maximum value from the dataset.

    Returns:
        float: Normalized variable.
    """
    if x == maximum:
        return 1
    return (x - minimum) / (maximum - minimum)


def get_allocation_scheme(
    n_gen: int, pop_size: int, sampling: str, cross: str, cross_prob: int, mutation: str, weights: int
) -> list:
    """Gets the allocation scheme used to drain servers during the maintenance using the genetic algorithm.

    Args:
        n_gen (int): Number of generations the genetic algorithm will run through.
        pop_size (int): Number of chromosomes that will represent the genetic algorithm's population.
        sampling (str): Sampling scheme used to create the initial population of the genetic algorithm.
        cross (str): Crossover method used to create the population offspring.
        cross_prob (int): Crossover probability.
        mutation (str): Mutation method used to ensure the population diversity across the generations.
        weights (int): ID of weighting configuration used to pick a solution from the Pareto-set.

    Returns:
        [list]: Placement scheme returned by the genetic algorithm.
    """
    # Defining the mutation probability
    services_hosted_by_outdated_servers = [service for service in Service.all() if not service.server.updated]
    mutation_prob = 1 / len(services_hosted_by_outdated_servers)

    # Defining genetic algorithm's attributes
    method = NSGA2(
        pop_size=pop_size,
        sampling=get_sampling(sampling),
        crossover=get_crossover(cross, prob=cross_prob),
        mutation=get_mutation(mutation, prob=mutation_prob),
        eliminate_duplicates=False,
    )

    # Running the genetic algorithm
    problem = PlacementProblem()
    res = minimize(problem, method, termination=("n_gen", n_gen), seed=1, verbose=VERBOSE, display=MyDisplay())

    # Parsing the genetic algorithm output
    solutions = []
    for i in range(len(res.X)):
        placement = res.X[i].tolist()
        fitness = {
            "Outd. Capacity": res.F[i][0],
            "Migr. Score": res.F[i][1],
            "SLA Violations": res.F[i][2],
        }
        overloaded_servers = res.CV[i][0].tolist()
        solutions.append({"placement": placement, "fitness": fitness, "overloaded_servers": overloaded_servers})

    weights = WEIGHT_OPTIONS[weights]

    # Gathering min and max values for each objective in the fitness function
    min_outdated_capacity = min([solution["fitness"]["Outd. Capacity"] for solution in solutions])
    max_outdated_capacity = max([solution["fitness"]["Outd. Capacity"] for solution in solutions])
    min_migration_score = min([solution["fitness"]["Migr. Score"] for solution in solutions])
    max_migration_score = max([solution["fitness"]["Migr. Score"] for solution in solutions])
    min_sla_violations = min([solution["fitness"]["SLA Violations"] for solution in solutions])
    max_sla_violations = max([solution["fitness"]["SLA Violations"] for solution in solutions])

    # Sorting solutions in the Pareto Set
    solutions = sorted(
        solutions,
        key=lambda s: (
            s["overloaded_servers"],
            min_max_norm(x=s["fitness"]["Outd. Capacity"], minimum=min_outdated_capacity, maximum=max_outdated_capacity)
            * weights[0]
            + min_max_norm(x=s["fitness"]["Migr. Score"], minimum=min_migration_score, maximum=max_migration_score)
            * weights[1]
            + min_max_norm(x=s["fitness"]["SLA Violations"], minimum=min_sla_violations, maximum=max_sla_violations)
            * weights[2],
        ),
    )

    print(f"{solutions[0]['fitness']}. Overloaded SVs: {solutions[0]['overloaded_servers']}. Weights: {weights}")

    return solutions[0]["placement"]


def laxus(arguments: dict):
    """Location-Aware Maintenance Algorithm

    Args:
        arguments (dict): List of arguments passed to the algorithm.
    """
    # Patching outdated edge servers hosting no services
    servers_to_patch = [server for server in EdgeServer.all() if not server.updated and len(server.services) == 0]
    if len(servers_to_patch) > 0:
        for server in servers_to_patch:
            server.update()

    # Migrating services to drain outdated edge servers
    else:
        allocation_scheme = get_allocation_scheme(
            n_gen=arguments["n_gen"],
            pop_size=arguments["pop_size"],
            sampling=arguments["sampling"],
            cross=arguments["cross"],
            cross_prob=arguments["cross_prob"],
            mutation=arguments["mutation"],
            weights=arguments["weights"],
        )

        services_hosted_by_outdated_servers = [service for service in Service.all() if not service.server.updated]

        for service_id, server_id in enumerate(allocation_scheme, 1):
            server = EdgeServer.instances[server_id - 1]
            service = services_hosted_by_outdated_servers[service_id - 1]

            if service.server != server:
                service.migrate(server)

                app = service.application
                user = app.users[0]
                user.set_communication_path(app)
