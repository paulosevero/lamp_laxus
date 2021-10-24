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
        servers_to_empty = [server for server in EdgeServer.all() if not server.updated]
        servers_to_empty = sorted(
            servers_to_empty, key=lambda sv: ((sv.patch + sv.sanity_check) * (1 / (sv.capacity + 1))) ** (1 / 2)
        )

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
                    sla = user.delay_slas[application]

                    # Sorting criteria: update status, violates SLA, occupation rate
                    candidate_servers = sorted(
                        candidate_servers,
                        key=lambda c: (
                            get_delay(user_base_station=user.base_station, origin=c, target=server),
                            -c.updated,
                            c.capacity - c.demand,
                        ),
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
