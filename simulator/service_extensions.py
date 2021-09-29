""""Contains a set of methods that extend services."""
# EdgeSimPy components
from edge_sim_py.components.service import Service

# Python libraries
import networkx as nx


def migrate(self, target_server: object, path: list = []) -> int:
    """Migrates the service to a target server.

    Args:
        target_server (object): Target server.

    Returns:
        migration_time (int): Service migration time.
    """
    # Finding a network path to migrate the service to the target host in case path is not provided
    if len(path) == 0 and self.server is not None:
        path = nx.shortest_path(
            G=self.simulator.topology,
            source=self.server.base_station,
            target=target_server.base_station,
            weight="bandwidth",
        )

    # Calculating service migration time
    migration_time = self.get_migration_time(path=path)

    # Storing migration metadata to post-simulation analysis
    self.migrations.append(
        {
            "batch": self.simulator.maintenance_batches,
            "duration": migration_time,
            "origin": self.server,
            "destination": target_server,
            "number_of_links_used": len(path) - 1,
        }
    )

    # Removing the service from its old host
    if self.server is not None:
        self.server.demand -= self.demand
        self.server.services.remove(self)

    # Adding the service to its new host
    self.server = target_server
    self.server.demand += self.demand
    self.server.services.append(self)

    return migration_time
