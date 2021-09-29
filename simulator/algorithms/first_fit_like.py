""" Contains a First-Fit-Like maintenance algorithm."""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer


def first_fit_like():
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

            for _ in range(len(server.services)):
                service = services.pop(0)

                # Migrating services using the First-Fit Decreasing heuristic, which suggests the
                # migration of services to the first server that has resources to host it
                for candidate_host in candidate_servers:
                    if candidate_host.capacity >= candidate_host.demand + service.demand:
                        # Migrating the service and storing the migration duration for post-simulation analysis
                        service.migrate(target_server=candidate_host)
                        break

            if len(server.services) == 0:
                servers_being_emptied.append(server)
