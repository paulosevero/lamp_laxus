""" Contains Location-Aware Maintenance Policy (LAMP) maintenance algorithm."""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer

# Python libraries
import networkx as nx


def get_delay(user_base_station: object, origin: object, target: object) -> int:
    """Gets the distance (in terms of delay) between two elements (origin and target).

    Args:
        user_base_station (object): Base station used by the user to access the edge network.
        origin (object): Origin object.
        target (object): Target object.

    Returns:
        int: Delay between origin and target.
    """
    topology = origin.simulator.topology

    path = nx.shortest_path(G=topology, source=origin.base_station, target=target.base_station, weight="delay")
    delay = topology.calculate_path_delay(path=path) + user_base_station.wireless_delay

    return delay


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


def get_service_delay_sla(service: object) -> int:
    """Gets the delay SLA of a given service.

    Args:
        service (object): Service object.

    Returns:
        int: Service's delay SLA.
    """
    application = service.application
    user = application.users[0]

    return user.delay_slas[application]


def get_server_delay_slas(server: object) -> list:
    """Gathers the list of delay SLAs of each service hosted by a given server.

    Args:
        server (object): Server object.

    Returns:
        list: List of delay SLAs of the services hosted by the server.
    """
    delay_slas = [get_service_delay_sla(service) for service in server.services]
    return delay_slas


def sort_servers_to_drain() -> list:
    """Sorts outdated servers given a set of criteria.

    Returns:
        sorted_outdated_servers [list]: Outdated servers.
    """
    outdated_servers = [server for server in EdgeServer.all() if not server.updated]

    min_capacity = min([1 / server.capacity for server in outdated_servers])
    max_capacity = max([1 / server.capacity for server in outdated_servers])

    min_demand = min([server.demand for server in outdated_servers])
    max_demand = max([server.demand for server in outdated_servers])

    min_update_duration = min([server.patch + server.sanity_check for server in outdated_servers])
    max_update_duration = max([server.patch + server.sanity_check for server in outdated_servers])

    for server in outdated_servers:
        norm_capacity_score = min_max_norm(
            x=1 / server.capacity,
            minimum=min_capacity,
            maximum=max_capacity,
        )
        norm_demand_score = min_max_norm(
            x=server.demand,
            minimum=min_demand,
            maximum=max_demand,
        )
        norm_update_duration_score = min_max_norm(
            x=server.patch + server.sanity_check,
            minimum=min_update_duration,
            maximum=max_update_duration,
        )

        server.drain_score = norm_capacity_score + norm_demand_score + norm_update_duration_score

    sorted_outdated_servers = sorted(outdated_servers, key=lambda server: server.drain_score)
    return sorted_outdated_servers


def sort_candidate_servers(user: object, server_being_drained: object, candidate_servers: list) -> list:
    min_free_resources = min([candidate.capacity - candidate.demand for candidate in candidate_servers])
    max_free_resources = max([candidate.capacity - candidate.demand for candidate in candidate_servers])
    min_delay = min(
        [
            get_delay(user_base_station=user.base_station, origin=server_being_drained, target=candidate)
            for candidate in candidate_servers
        ]
    )
    max_delay = max(
        [
            get_delay(user_base_station=user.base_station, origin=server_being_drained, target=candidate)
            for candidate in candidate_servers
        ]
    )

    for candidate in candidate_servers:
        update_score = 0 if candidate.updated else 1
        norm_free_resources_score = min_max_norm(
            x=candidate.capacity - candidate.demand,
            minimum=min_free_resources,
            maximum=max_free_resources,
        )
        norm_delay_score = min_max_norm(
            x=get_delay(user_base_station=user.base_station, origin=server_being_drained, target=candidate),
            minimum=min_delay,
            maximum=max_delay,
        )

        candidate.score = update_score + norm_free_resources_score + norm_delay_score

    candidate_servers = sorted(candidate_servers, key=lambda candidate: candidate.score)
    return candidate_servers


def lamp(arguments: dict):
    # Patching outdated edge servers hosting no services
    servers_to_patch = [server for server in EdgeServer.all() if not server.updated and len(server.services) == 0]
    if len(servers_to_patch) > 0:
        for server in servers_to_patch:
            server.update()

    # Migrating services to drain outdated edge servers
    else:
        servers_being_emptied = []

        # Getting the list of servers that still need to be patched
        servers_to_empty = sort_servers_to_drain()

        for server in servers_to_empty:
            # We consider as candidate hosts for the services all EdgeServer
            # objects not being emptied in the current maintenance step
            candidate_servers = [
                candidate
                for candidate in EdgeServer.all()
                if candidate not in servers_being_emptied and candidate != server
            ]

            # Sorting services by its demand (decreasing)
            services = sorted(list(server.services), key=lambda service: -service.demand)

            if EdgeServer.can_host_services(servers=candidate_servers, services=services):
                for _ in range(len(server.services)):
                    service = services.pop(0)
                    application = service.application
                    user = application.users[0]

                    # Sorting candidate servers to host the service
                    candidate_servers = sort_candidate_servers(
                        user=user, server_being_drained=server, candidate_servers=candidate_servers
                    )

                    for candidate_host in candidate_servers:
                        if candidate_host.capacity >= candidate_host.demand + service.demand:
                            # Migrating the service and storing the migration duration for post-simulation analysis
                            service.migrate(target_server=candidate_host)

                            # Redefining the set of links used to communicate the user to his service
                            app = service.application
                            user = app.users[0]
                            user.set_communication_path(app)

                            break

            if len(server.services) == 0:
                servers_being_emptied.append(server)
