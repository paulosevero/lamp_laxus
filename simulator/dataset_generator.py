"""Contains a set of methods used to create datasets for the simulator.

Objects are created using a group of "builders",
classes that implement the Builder design pattern to instantiate objects with different properties in an organized way.
More information about the Builder design pattern can be found in the links below:
- https://refactoring.guru/design-patterns/builder
- https://refactoring.guru/design-patterns/builder/python/example
"""
# Python libraries
import random
import json
import networkx as nx

# EdgeSimPy components
from edge_sim_py.components.base_station import BaseStation
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.application import Application
from edge_sim_py.components.service import Service
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.user import User

# Helper builders
from edge_sim_py.component_builders.map_builder import create_hexagonal_grid
from edge_sim_py.component_builders.distributions_builder import uniform

# Component builders
from edge_sim_py.component_builders.base_station_builder import BaseStationBuilder
from edge_sim_py.component_builders.edge_server_builder import EdgeServerBuilder
from edge_sim_py.component_builders.application_builder import ApplicationBuilder
from edge_sim_py.component_builders.service_builder import ServiceBuilder
from edge_sim_py.component_builders.user_builder import UserBuilder


def find_neighbors_hexagonal_grid(map_coordinates: list, current_position: tuple) -> list:
    """Finds the set of adjacent positions of coordinates 'current_position' in a hexagonal grid.

    Args:
        map_coordinates (list): List of map coordinates.
        current_position (tuple): Current position of the hexagonal grid whose neighbors we want to find.

    Returns:
        neighbors (list): List of neighbors from the current position in the hexagonal grid map.
    """
    x = current_position[0]
    y = current_position[1]

    candidates = [(x - 2, y), (x - 1, y + 1), (x + 1, y + 1), (x + 2, y), (x + 1, y - 1), (x - 1, y - 1)]

    neighbors = [
        neighbor
        for neighbor in candidates
        if neighbor[0] >= 0 and neighbor[1] >= 0 and (neighbor[0], neighbor[1]) in map_coordinates
    ]

    return neighbors


def closest_fit():
    """Migrates the list of services to the edge servers located closer to its users in the map."""
    topology = Topology.first()
    services = random.sample(Service.all(), Service.count())

    for service in services:
        app = service.application
        user = app.users[0]

        host_candidates = []
        for edge_server in EdgeServer.all():
            # Computing the shortest path between the client's base station and the current edge server's base station
            path = nx.shortest_path(
                G=topology,
                source=user.base_station,
                target=edge_server.base_station,
                weight=lambda u, v, d: topology[u][v]["delay"]
                if topology[u][v]["demand"] + app.network_demand <= topology[u][v]["bandwidth"]
                else topology[u][v]["delay"] * 10,
                method="dijkstra",
            )

            # Computing the shortest path delay
            delay = topology.calculate_path_delay(path)

            # Adding the current EdgeServer to the list of host candidates to host application's services
            host_candidates.append({"obj": edge_server, "delay": delay})

        host_candidates = sorted(host_candidates, key=lambda n: n["delay"])

        for edge_server_data in host_candidates:
            edge_server = edge_server_data["obj"]

            # If the closest edge node available is not being used to host the service,
            # checks if it handles the service demand. If positive, migrates the service to it
            if edge_server.capacity - edge_server.demand >= service.demand:
                edge_server.services.append(service)
                edge_server.demand += service.demand
                service.server = edge_server
                break


def set_update_status_all_edge_servers(self, update_statuses: list) -> list:
    """Defines the update status of edge server objects.
    Args:
        update_statuses (list): Update statuses assigned to the edge server objects.
    Returns:
        edge_servers (list): Modified EdgeServer objects.
    """
    for index, edge_server in enumerate(self.objects):
        edge_server.updated = update_statuses[index]

    return self.objects


def set_patches_all_edge_servers(self, patch_values: list) -> list:
    """Defines the time it takes to update edge servers.
    Args:
        patch_values (list): Patch durations assigned to the edge server objects.
    Returns:
        edge_servers (list): Modified EdgeServer objects.
    """
    for index, edge_server in enumerate(self.objects):
        edge_server.patch = patch_values[index]

    return self.objects


def set_sanity_checks_all_edge_servers(self, sanity_check_values: list) -> list:
    """Defines the time it takes to run sanity checks on edge servers after they are updated.
    Args:
        sanity_check_values (list): Sanity check durations assigned to the edge server objects.
    Returns:
        edge_servers (list): Modified EdgeServer objects.
    """
    for index, edge_server in enumerate(self.objects):
        edge_server.sanity_check = sanity_check_values[index]

    return self.objects


# Adjusting the level of verbosity of the dataset generator
VERBOSE = False

# Defining seed values to enable reproducibility
SEED = 1
random.seed(SEED)


# Adding/Overriding dataset generator methods
EdgeServerBuilder.set_update_status_all_edge_servers = set_update_status_all_edge_servers
EdgeServerBuilder.set_patches_all_edge_servers = set_patches_all_edge_servers
EdgeServerBuilder.set_sanity_checks_all_edge_servers = set_sanity_checks_all_edge_servers


# Defining number of simulation steps
simulation_steps = 1

# Creating list of hexagons to represent the map
map_coordinates = create_hexagonal_grid(x_size=10, y_size=10)

# Creating base stations
n_base_stations = len(map_coordinates)
base_station_builder = BaseStationBuilder()
base_station_builder.create_objects(n_objects=n_base_stations)
base_station_builder.set_coordinates_all_base_stations(coordinates=map_coordinates)
base_station_wireless_delays = uniform(n_items=n_base_stations, valid_values=[15], shuffle_distribution=True)
base_station_builder.set_wireless_delay_all_base_stations(wireless_delay_values=base_station_wireless_delays)


# Creating edge servers
n_edge_servers = 40
edge_server_builder = EdgeServerBuilder()
edge_server_builder.create_objects(n_objects=n_edge_servers)
edge_servers_coordinates = random.sample(map_coordinates, n_edge_servers)
edge_server_builder.set_coordinates_all_edge_servers(coordinates=edge_servers_coordinates)
edge_servers_capacity = uniform(n_items=n_edge_servers, valid_values=[200, 250], shuffle_distribution=True)
edge_server_builder.set_capacity_all_edge_servers(capacity_values=edge_servers_capacity)

edge_servers_update_statuses = uniform(n_items=n_edge_servers, valid_values=[False], shuffle_distribution=True)
edge_server_builder.set_update_status_all_edge_servers(update_statuses=edge_servers_update_statuses)
edge_servers_patches = uniform(n_items=n_edge_servers, valid_values=[250, 350], shuffle_distribution=True)
edge_server_builder.set_patches_all_edge_servers(patch_values=edge_servers_patches)
edge_servers_sanity_checks = uniform(n_items=n_edge_servers, valid_values=[300, 400], shuffle_distribution=True)
edge_server_builder.set_sanity_checks_all_edge_servers(sanity_check_values=edge_servers_sanity_checks)


# Creating applications and services (and defining relationships between them)
n_applications = 90
application_builder = ApplicationBuilder()
application_builder.create_objects(n_objects=n_applications)
network_demands = uniform(n_items=n_applications, valid_values=[1, 2], shuffle_distribution=True)
application_builder.set_network_demand_all_applications(network_demands=network_demands)

services_per_application = uniform(n_items=n_applications, valid_values=[1], shuffle_distribution=True)

n_services = sum(services_per_application)
service_builder = ServiceBuilder()
service_builder.create_objects(n_objects=n_services)
service_demands = uniform(n_items=n_services, valid_values=[20, 40, 60], shuffle_distribution=True)
service_builder.set_demand_all_services(demand_values=service_demands)

for index, application in enumerate(Application.all()):
    for _ in range(services_per_application[index]):
        service = next((service for service in Service.all() if service.application is None), None)
        if service is not None:
            service.application = application
            application.services.append(service)


# Creating network topology
network_nodes = BaseStation.all()
topology_name = "Barabási-Albert"

if topology_name == "Barabási-Albert":
    topology = Topology.new_barabasi_albert(nodes=network_nodes, seed=SEED, delay=10, bandwidth=5, min_links_per_node=1)
elif topology_name == "Partially Connected Mesh":
    # Manually defining a network topology based on https://doi.org/10.1145/3344341.3368818,
    # where nodes represent a hexagonal grid, and each node has a link to all its direct
    # neighbors (i.e., central node have 6 links) connecting them to other nodes

    # Creating BetworkX object
    topology = Topology()

    # Defining network nodes
    topology.add_nodes_from(network_nodes)

    # Adding links to each network node
    for base_station in BaseStation.all():
        neighbor_coordinates = find_neighbors_hexagonal_grid(
            map_coordinates=map_coordinates, current_position=base_station.coordinates
        )

        for coordinates in neighbor_coordinates:
            neighbor_base_station = BaseStation.find_by("coordinates", coordinates)
            topology.add_edge(base_station, neighbor_base_station)


# Defining link attributes
n_links = len(list(topology.edges))
link_bandwidths = uniform(n_items=n_links, valid_values=[5, 10], shuffle_distribution=True)
link_delays = uniform(n_items=n_links, valid_values=[5, 10], shuffle_distribution=True)

for link in topology.edges(data=True):
    topology[link[0]][link[1]]["bandwidth"] = link_bandwidths[0]
    topology[link[0]][link[1]]["delay"] = link_delays[0]

    # Updating attribute lists after the link is updated
    link_bandwidths.pop(0)
    link_delays.pop(0)


# Creating users
n_users = n_applications
user_builder = UserBuilder()
user_builder.create_objects(n_objects=n_users)
user_builder.set_target_positions(map_coordinates=map_coordinates, n_target_positions=20)
user_builder.set_pathway_mobility_all_users(
    map_coordinates=map_coordinates, steps=simulation_steps, target_positions=False
)
users_per_application = uniform(n_items=n_users, valid_values=[1], shuffle_distribution=True)
for index, user in enumerate(User.all()):
    delay_slas = uniform(n_items=users_per_application[index], valid_values=[45, 90], shuffle_distribution=True)

    for i in range(users_per_application[index]):
        application = next((application for application in Application.all() if len(application.users) <= i), None)
        if application is not None:
            application.users.append(user)
            user.applications.append(application)
            user.delay_slas[application] = delay_slas[i]


# Defining the initial service placement scheme
closest_fit()


# Updating users communication paths
for user in User.all():
    for application in user.applications:
        user.communication_paths[application] = []
        communication_chain = [user] + application.services

        # Defining a set of links to connect the items in the application's service chain
        for j in range(len(communication_chain) - 1):

            # Defining origin and target nodes
            origin = user.base_station if communication_chain[j] == user else communication_chain[j].server.base_station
            target = (
                user.base_station
                if communication_chain[j + 1] == user
                else communication_chain[j + 1].server.base_station
            )
            # Finding the best communication path
            path = Topology.first().get_shortest_path(
                origin=origin,
                target=target,
                user=user,
                app=application,
            )
            # Adding the best path found to the communication path
            user.communication_paths[application].extend(path)

        # Removing duplicated entries in the communication path to avoid NetworkX crashes
        user.communication_paths[application] = Topology.first().remove_path_duplicates(
            path=user.communication_paths[application]
        )

        # Computing the new demand of chosen links
        Topology.first().allocate_communication_path(
            communication_path=user.communication_paths[application], app=application
        )

        # The initial application delay is given by the time it takes to communicate its client and his base station
        delay = user.base_station.wireless_delay

        # Adding the communication path delay to the application's delay
        communication_path = user.communication_paths[application]
        delay += Topology.first().calculate_path_delay(path=communication_path)

        # Updating application delay inside user's 'applications' attribute
        user.delays[application] = delay


if VERBOSE:
    print("\nBase Stations:")
    for base_station in BaseStation.all():
        print(
            f"    {base_station}. Coordinates: {base_station.coordinates}. Wireless Delay: {base_station.wireless_delay}"
        )

    print("\n\nEdge Servers:")
    for edge_server in EdgeServer.all():
        print(
            f"    {edge_server}. Coordinates: {edge_server.coordinates}. Capacity: {edge_server.capacity}. Base Station: {edge_server.base_station} ({edge_server.base_station.coordinates})"
        )

    print("\n\nApplications:")
    for application in Application.all():
        print(f"    {application}. Network Demand: {application.network_demand}.")
        for service in application.services:
            print(f"        {service}. Demand: {service.demand}. Server: {service.server}")

    print("\n\nUsers:")
    for user in User.all():
        print(
            f"    {user}. Coordinates: {user.coordinates}. Base Station: {user.base_station} ({user.base_station.coordinates})"
        )

        for app in user.applications:
            print(
                f"        {app}. SLA: {user.delay_slas[app]}. Delay {user.delays[app]}. Communication Path: {user.communication_paths[app]}"
            )


# Creating a dictionary that will be converted to a JSON object containing the dataset
dataset = {}


# General dataset information
dataset["simulation_steps"] = simulation_steps
dataset["coordinates_system"] = "hexagonal_grid"

dataset["base_stations"] = [
    {
        "id": base_station.id,
        "coordinates": base_station.coordinates,
        "wireless_delay": base_station.wireless_delay,
        "users": [user.id for user in base_station.users],
        "edge_servers": [edge_server.id for edge_server in base_station.edge_servers],
    }
    for base_station in BaseStation.all()
]

dataset["edge_servers"] = [
    {
        "id": edge_server.id,
        "capacity": edge_server.capacity,
        "base_station": edge_server.base_station.id,
        "coordinates": edge_server.coordinates,
        "updated": edge_server.updated,
        "patch": edge_server.patch,
        "sanity_check": edge_server.sanity_check,
        "services": [service.id for service in edge_server.services],
    }
    for edge_server in EdgeServer.all()
]

dataset["users"] = [
    {
        "id": user.id,
        "base_station": {"type": "BaseStation", "id": user.base_station.id},
        "applications": [
            {
                "id": app.id,
                "delay_sla": user.delay_slas[app],
                "communication_path": [
                    {"type": "BaseStation", "id": base_station.id} for base_station in user.communication_paths[app]
                ],
            }
            for app in user.applications
        ],
        "coordinates_trace": user.coordinates_trace,
    }
    for user in User.all()
]

dataset["applications"] = [
    {
        "id": application.id,
        "network_demand": application.network_demand,
        "services": [service.id for service in application.services],
        "users": [user.id for user in application.users],
    }
    for application in Application.all()
]

dataset["services"] = [
    {
        "id": service.id,
        "demand": service.demand,
        "server": {"type": "EdgeServer", "id": service.server.id if service.server else None},
        "application": service.application.id,
    }
    for service in Service.all()
]

network_links = []
for index, link in enumerate(Topology.first().edges(data=True)):
    nodes = [
        {"type": "BaseStation", "id": link[0].id},
        {"type": "BaseStation", "id": link[1].id},
    ]
    delay = Topology.first()[link[0]][link[1]]["delay"]
    bandwidth = Topology.first()[link[0]][link[1]]["bandwidth"]
    network_links.append({"id": index + 1, "nodes": nodes, "delay": delay, "bandwidth": bandwidth})

dataset["network"] = {"links": network_links}

# Defining output file name
dataset_file_name = "dataset1"

# Storing the dataset to an output file
with open(f"datasets/{dataset_file_name}.json", "w") as output_file:
    json.dump(dataset, output_file, indent=4)
