""""Contains a set of methods that extend edge servers with maintenance-related features."""
# EdgeSimPy components
from edge_sim_py.components.edge_server import EdgeServer


def outdated(self=None) -> list:
    """Gathers the list of servers that were not updated yet.

    Returns:
        outdated_servers (list): Group of outdated edge servers.
    """
    outdated_servers = [edge_server for edge_server in EdgeServer.all() if not edge_server.updated]
    return outdated_servers


def updated(self=None) -> list:
    """Gathers the list of updated servers.

    Returns:
        outdated_servers (list): Group of updated edge servers.
    """
    updated_servers = [edge_server for edge_server in EdgeServer.all() if edge_server.updated]
    return updated_servers


def ready_to_update(self=None) -> list:
    """Gathers the list of edge servers ready to be updated. An edge server is ready to update when it
    fulfills two conditions: (i) It must be outdated and (ii) It must be empty (i.e., not hosting services).

    Returns:
        servers_ready_to_update (list): Group of edge servers that can be updated immediately.
    """
    servers_ready_to_update = [edge_server for edge_server in EdgeServer.outdated() if len(edge_server.services) == 0]
    return servers_ready_to_update


def can_host_services(servers: list, services: list) -> bool:
    """Checks if a set of servers have resources to host a group of services.

    Args:
        servers (list): List of edge servers.
        services (list): List of services that we want to accommodate inside the servers.

    Returns:
        bool: Boolean expression that tells us whether the set of servers did manage or not to host the services.
    """
    services_allocated = 0

    # Sorting services by demand (in decreasing order) to avoid resource wastage
    services = sorted(services, key=lambda service: -service.demand)

    # Checking if all services could be hosted by the list of servers
    for service in services:
        # Sorting servers according to their demand (descending)
        servers = sorted(servers, key=lambda sv: sv.capacity - sv.demand)
        for server in servers:
            if server.capacity >= server.demand + service.demand:
                server.demand += service.demand
                services_allocated += 1
                break

    # Recomputing servers' demand
    for server in servers:
        server.compute_demand()

    return len(services) == services_allocated


def update(self) -> int:
    """Updates the edge server.

    Returns:
        int: Update duration.
    """
    self.updated = True

    update_duration = self.patch + self.sanity_check

    # Storing update metadata
    self.update_metadata = {
        "maintenance_batch": self.simulator.maintenance_batches,
        "duration": update_duration,
    }

    return update_duration
