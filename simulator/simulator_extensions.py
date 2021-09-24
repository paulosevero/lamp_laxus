"""Contains helper methods that extend EdgeSimPy to simulate maintenance in edge infrastructures.
"""
# EdgeSimPy components
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.base_station import BaseStation
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.user import User
from edge_sim_py.components.application import Application
from edge_sim_py.components.service import Service

# Python libraries
import json


def load_maintenance_attributes(self, input_file: str):
    """Loads maintenance-specific parameters from a dataset file.

    Args:
        input_file (str): Dataset file name.
    """
    # Adding 'Simulator' reference inside EdgeSimPy components
    objects = Topology.all() + BaseStation.all() + EdgeServer.all() + User.all() + Application.all() + Service.all()
    for obj in objects:
        obj.simulator = self

    with open(input_file, "r") as read_file:
        data = json.load(read_file)

    # Extending EdgeServer objects (adding patches and sanity checks)
    for server in EdgeServer.all():
        server_data = next((obj_data for obj_data in data["edge_servers"] if obj_data["id"] == server.id), None)

        if server_data is not None:
            server.updated = server_data["updated"]
            server.patch = server_data["patch"]
            server.sanity_check = server_data["sanity_check"]


def show_simulated_environment():
    print("Edge Servers:")
    for server in EdgeServer.all():
        print(
            f"    {server}. Cap/Dem: [{server.capacity}, {server.demand}]. Updated: {server.updated}. Patch: {server.patch}. Sanity check: {server.sanity_check}. Base Station: {server.base_station.id}"
        )

    print("Users:")
    for user in User.all():
        app = user.applications[0]
        sla = user.delay_slas[app]
        user.set_communication_path(app=app)
        print(
            f"    {user}. Location: {user.coordinates}. Base Station: {user.base_station.id} ({user.base_station.coordinates}). App: {app.id}. SLA: {sla}"
        )

    print("Applications:")
    for app in Application.all():
        user = app.users[0]
        service = app.services[0]
        delay = user.delays[app]
        print(
            f"    {app}. Service: {service} ({service.demand}). Comm Path: {user.communication_paths[app]}. Delay: {delay}"
        )
