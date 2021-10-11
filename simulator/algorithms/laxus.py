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
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_reference_directions

ALGORITHM = "NSGA-II"
VERBOSE = True
PARALLEL = False
N_THREADS = 4


def min_max_norm(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)


class MyDisplay(Display):
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)

        outdated_servers_occupied = int(np.min(algorithm.pop.get("F")[:, 0]))
        useless_migrations = int(np.min(algorithm.pop.get("F")[:, 1]))
        sla_violations = int(np.min(algorithm.pop.get("F")[:, 2]))

        self.output.append("Out.SVs Used", outdated_servers_occupied)
        self.output.append("Useless Migr", useless_migrations)
        self.output.append("SLA viol.", sla_violations)


class PlacementProblem(Problem):
    def __init__(self, **kwargs):
        services_on_outdated_hosts = len([1 for service in Service.all() if not service.server.updated])
        super().__init__(
            n_var=services_on_outdated_hosts, n_obj=3, n_constr=1, xl=1, xu=EdgeServer.count(), type_var=int, **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        if PARALLEL:
            thread_pool = ThreadPool(N_THREADS)
            output = thread_pool.map(self.get_fitness_score_and_constraints, x)
            thread_pool.close()
        else:
            output = [self.get_fitness_score_and_constraints(solution=solution) for solution in x]

        out["F"] = np.array([item[0] for item in output])
        out["G"] = np.array([item[1] for item in output])

    def get_fitness_score_and_constraints(self, solution: list) -> tuple:
        ##########################################
        ## Calculating objectives and penalties ##
        ##########################################
        # Declaring objectives
        sla_violations = 0
        useless_migrations = 0
        migrations_to_outdated_servers = 0

        # Objective 1: minimize the number of outdated servers hosting applications
        outdated_servers_occupied = sum([1 for server in EdgeServer.outdated() if server.id in solution])

        # Gathering edge servers capacity
        edge_servers_capacity = [edge_server.capacity for edge_server in EdgeServer.all()]

        services_hosted_by_outdated_servers = [service for service in Service.all() if not service.server.updated]

        for service_id, server_id in enumerate(solution, 1):
            server = EdgeServer.instances[server_id - 1]
            service = services_hosted_by_outdated_servers[service_id - 1]
            user = service.application.users[0]

            # Updating edge server occupation
            edge_servers_capacity[server.id - 1] -= service.demand

            # Counting the number of useless migrations that don't lead an outdated server drained
            if service.server != server and service.server.id in solution:
                useless_migrations += 1

            # Objective 3: minimize the number of SLA violations
            sla = user.delay_slas[service.application]
            topology = Topology.first()
            shortest_path = nx.shortest_path(
                G=topology,
                source=user.base_station,
                target=server.base_station,
                weight=lambda u, v, d: topology[u][v]["delay"]
                if topology[u][v]["demand"] + service.application.network_demand <= topology[u][v]["bandwidth"]
                else topology[u][v]["delay"] * 10,
                method="dijkstra",
            )

            # Counting the number of applications migrated to outdated servers
            if service.server != server and not server.updated:
                migrations_to_outdated_servers += 1

            delay = user.base_station.wireless_delay + topology.calculate_path_delay(shortest_path)
            if delay > sla:
                sla_violations += 1

        # Objective 2: minimize the number of undesired migrations
        undesired_migrations = useless_migrations + migrations_to_outdated_servers

        fitness = (outdated_servers_occupied, undesired_migrations, sla_violations)
        overloaded_servers = sum([1 for item in edge_servers_capacity if item < 0])

        return (fitness, overloaded_servers)


def get_allocation_scheme(n_gen: int, pop_size: int, sampling: str, cross: str, cross_prob: int, mutation: str) -> list:
    if ALGORITHM == "NSGA-II":
        method = NSGA2(
            pop_size=pop_size,
            sampling=get_sampling(sampling),
            crossover=get_crossover(cross, prob=cross_prob),
            mutation=get_mutation(mutation, prob=1 / Service.count()),
            eliminate_duplicates=False,
        )
    elif ALGORITHM == "NSGA-III":
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=3)
        method = NSGA3(
            pop_size=pop_size,
            ref_dirs=ref_dirs,
            sampling=get_sampling(sampling),
            crossover=get_crossover(cross, prob=cross_prob),
            mutation=get_mutation(mutation, prob=1 / Service.count()),
            eliminate_duplicates=False,
        )

    problem = PlacementProblem()
    res = minimize(problem, method, termination=("n_gen", n_gen), seed=1, verbose=VERBOSE, display=MyDisplay())

    solutions = []
    for i in range(len(res.X)):
        placement = res.X[i].tolist()
        fitness = {
            "Outd. Servers Used": res.F[i][0],
            "Useless Migrations": res.F[i][1],
            "SLA Violations": res.F[i][2],
        }
        overloaded_servers = res.CV[i][0].tolist()
        solutions.append({"placement": placement, "fitness": fitness, "overloaded_servers": overloaded_servers})

    solutions = sorted(
        solutions,
        key=lambda s: (
            s["overloaded_servers"],
            min_max_norm(x=s["fitness"]["Outd. Servers Used"], minimum=0, maximum=len(EdgeServer.outdated()))
            + min_max_norm(x=s["fitness"]["Useless Migrations"], minimum=0, maximum=Service.count())
            + min_max_norm(x=s["fitness"]["SLA Violations"], minimum=0, maximum=Service.count()),
        ),
    )

    print(f"SOLUTION: {solutions[0]['fitness']}. {solutions[0]['overloaded_servers']}")

    return solutions[0]["placement"]


def laxus(arguments: dict):
    # Patching outdated edge servers hosting no services
    servers_to_patch = [server for server in EdgeServer.all() if not server.updated and len(server.services) == 0]
    if len(servers_to_patch) > 0:
        for server in servers_to_patch:
            server.update()

    # Migrating services to drain outdated edge servers
    else:
        outdated_servers_hosting_apps = sum([1 for server in EdgeServer.outdated() if len(server.services) > 0])
        print(f"[BEFORE] OUTDATED SERVERS HOSTING APPLICATIONS: {outdated_servers_hosting_apps}")

        allocation_scheme = get_allocation_scheme(
            n_gen=arguments["n_gen"],
            pop_size=arguments["pop_size"],
            sampling=arguments["sampling"],
            cross=arguments["cross"],
            cross_prob=arguments["cross_prob"],
            mutation=arguments["mutation"],
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

        outdated_servers_hosting_apps = sum([1 for server in EdgeServer.outdated() if len(server.services) > 0])
        print(f"[AFTER] OUTDATED SERVERS HOSTING APPLICATIONS: {outdated_servers_hosting_apps}")
