"""Contains helper methods that extend EdgeSimPy to simulate maintenance in edge infrastructures.
"""
# EdgeSimPy components
from edge_sim_py.components.topology import Topology
from edge_sim_py.components.base_station import BaseStation
from edge_sim_py.components.edge_server import EdgeServer
from edge_sim_py.components.user import User
from edge_sim_py.components.application import Application
from edge_sim_py.components.service import Service

# Maintenance-related methods
from simulator.edge_server_extensions import outdated
from simulator.edge_server_extensions import updated
from simulator.edge_server_extensions import ready_to_update
from simulator.edge_server_extensions import can_host_services
from simulator.edge_server_extensions import update
from simulator.service_extensions import migrate

# Python libraries
import json
import typing
import time


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
            server.update_metadata = {}

    # Adding new methods to EdgeServer objects
    EdgeServer.outdated = outdated
    EdgeServer.updated = updated
    EdgeServer.ready_to_update = ready_to_update
    EdgeServer.can_host_services = can_host_services
    EdgeServer.update = update

    # Adding new methods to Service objects
    Service.migrate = migrate


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


def update_state(self, step: int):
    """Updates the system state.

    Args:
        step (int): Current simulation time step.
    """
    self.current_step = step


def run(self, algorithm: typing.Callable, arguments: list = []):
    """Executes the simulation.

    Args:
        algorithm (typing.Callable): Algorithm that will be executed during simulation.
    """
    self.set_simulator_attribute_inside_objects()

    # Adding a reference to the network topology inside the Simulator instance
    self.topology = Topology.first()

    # Creating an empty list to accommodate the simulation metrics
    algorithm_name = f"{str(algorithm).split(' ')[1]}-{time.time()}"
    self.metrics[algorithm_name] = []

    # Storing original objects state
    self.store_original_state()

    # Resetting simulation time attributes
    self.current_step = 0
    self.maintenance_batches = 0

    # The simulation continues until all servers are updated
    while len(EdgeServer.outdated()) > 0:
        # Incrementing the number of maintenance batches
        self.maintenance_batches += 1

        # Updating system state according to the new simulation time step
        self.update_state(step=self.current_step + 1)

        # Executing user-specified algorithm
        algorithm(arguments=arguments)

        # Collecting metrics for the current simulation step
        self.collect_metrics(algorithm=algorithm_name)

    # Restoring original objects state
    self.restore_original_state()


def store_original_state(self):
    """Stores the original state of all objects in the simulator."""
    # Edge servers update status
    self.original_system_state["edge_servers"] = {}
    for edge_server in EdgeServer.all():
        self.original_system_state["edge_servers"][edge_server] = {"updated": edge_server.updated}

    # Services placement
    self.original_system_state["services"] = {}
    for service in Service.all():
        self.original_system_state["services"][service] = {"server": service.server}

    # Users locations and applications routing
    self.original_system_state["users"] = {}
    for user in User.all():
        self.original_system_state["users"][user] = {
            "base_station": user.base_station,
            "communication_paths": user.communication_paths.copy(),
        }


def restore_original_state(self):
    """Restores the original state of all objects in the simulator."""
    # Edge servers update status
    for edge_server in EdgeServer.all():
        edge_server.updated = self.original_system_state["edge_servers"][edge_server]["updated"]

    # Services placement
    for service in Service.all():
        server = self.original_system_state["services"][service]["server"]
        if server is not None:
            service.migrate(target_server=server)
        service.migrations = []

    # Users locations and applications routing
    for user in User.all():
        user.coordinates = user.coordinates_trace[0]
        user.base_station = self.original_system_state["users"][user]["base_station"]
        for application in user.applications:
            user.set_communication_path(
                app=application,
                communication_path=self.original_system_state["users"][user]["communication_paths"][application],
            )


def collect_metrics(self, algorithm: str):
    """Collects simulation metrics.

    Args:
        algorithm (str): Name of the algorithm being executed.
    """

    # Collecting generic metrics
    current_batch_duration = 0

    for service in Service.all():
        for migration in service.migrations:
            if migration["batch"] == self.maintenance_batches:
                current_batch_duration += migration["duration"]

    overloaded_servers = 0
    updates_in_the_current_batch = []
    for edge_server in EdgeServer.all():

        if edge_server.demand > edge_server.capacity:
            overloaded_servers += 1

        if (
            edge_server.updated
            and "maintenance_batch" in edge_server.update_metadata
            and edge_server.update_metadata["maintenance_batch"] == self.maintenance_batches
        ):
            updates_in_the_current_batch.append(edge_server.update_metadata["duration"])

    if len(updates_in_the_current_batch) > 0:
        current_batch_duration = max(updates_in_the_current_batch)

    # Calculating the overall maintenance duration (time elapsed in the current and previous batches)
    previous_batches_duration = sum(
        previous_batch_metrics["batch_duration"] for previous_batch_metrics in self.metrics[algorithm]
    )
    maintenance_duration = previous_batches_duration + current_batch_duration

    # Collecting edgeServer-related metrics
    used_servers = 0
    updated_servers = 0
    outdated_servers = 0
    for edge_server in EdgeServer.all():
        # Number of servers used to accommodate services
        if len(edge_server.services) > 0:
            used_servers += 1

        # Number of outdated and updated servers
        if edge_server.updated:
            updated_servers += 1
        else:
            outdated_servers += 1

    # Vulnerability surface
    vulnerability_surface = outdated_servers * maintenance_duration

    # Collecting application-related metrics
    sla_violations = 0
    safeguarded_services = 0
    vulnerable_services = 0
    migrations = []
    for application in Application.all():
        user = application.users[0]
        user.set_communication_path(application)

        # Number of SLA Violations
        if user.delays[application] > user.delay_slas[application]:
            sla_violations += 1

        for service in application.services:
            # Number of safeguarded and vulnerable services. Servers are
            # safeguarded when their servers are updated and vulnerable otherwise)
            if service.server.updated:
                safeguarded_services += 1
            else:
                vulnerable_services += 1

            for migration in service.migrations:
                if migration["batch"] == self.maintenance_batches:
                    migrations.append(migration["duration"])

        if len(migrations) > 0:
            overall_migration_duration = sum(migrations)
            longest_migration_duration = max(migrations)
            average_migration_duration = sum(migrations) / len(migrations)
        else:
            overall_migration_duration = 0
            longest_migration_duration = 0
            average_migration_duration = 0

    # Creating the structure to accommodate simulation metrics for the current maintenance batch
    self.metrics[algorithm].append(
        {
            "batch": self.maintenance_batches,
            "batch_duration": current_batch_duration,
            "overall_maintenance_duration": maintenance_duration,
            "overloaded_servers": overloaded_servers,
            "used_servers": used_servers,
            "updated_servers": updated_servers,
            "outdated_servers": outdated_servers,
            "vulnerability_surface": vulnerability_surface,
            "sla_violations": sla_violations,
            "safeguarded_services": safeguarded_services,
            "vulnerable_services": vulnerable_services,
            "migrations": migrations,
            "overall_migration_duration": overall_migration_duration,
            "average_migration_duration": average_migration_duration,
            "longest_migration_duration": longest_migration_duration,
        }
    )


def show_results(self, verbosity: bool):
    """Displays the simulation results.

    Args:
        verbosity (bool): Controls the output verbosity.
    """
    for algorithm, results in self.metrics.items():
        overloaded_servers = 0
        consolidation_rate = []
        vulnerability_surface = 0
        sla_violations = 0
        number_of_migrations = 0
        migrations = []
        overall_migration_duration = 0
        average_migration_duration = []
        longest_migration_duration = 0

        updated_servers_per_batch = []
        outdated_servers_per_batch = []
        safeguarded_services_per_batch = []
        vulnerable_services_per_batch = []
        maintenance_duration_per_batch = []
        sla_violations_per_batch = []
        migrations_duration_per_batch = []

        print(f"\nAlgorithm: {algorithm}")
        for batch_results in results:
            consolidation_rate.append(100 - (batch_results["used_servers"] * 100 / EdgeServer.count()))
            vulnerability_surface += batch_results["vulnerability_surface"]
            sla_violations += batch_results["sla_violations"]
            number_of_migrations += len(batch_results["migrations"])
            migrations.extend(batch_results["migrations"])
            migrations_duration_per_batch.append(sum([migration for migration in batch_results["migrations"]]))
            overall_migration_duration += batch_results["overall_migration_duration"]
            average_migration_duration.append(batch_results["average_migration_duration"])
            if longest_migration_duration < batch_results["longest_migration_duration"]:
                longest_migration_duration = batch_results["longest_migration_duration"]

            overloaded_servers += batch_results["overloaded_servers"]
            maintenance_duration_per_batch.append(int(batch_results["overall_maintenance_duration"]))
            updated_servers_per_batch.append(batch_results["updated_servers"])
            outdated_servers_per_batch.append(batch_results["outdated_servers"])
            sla_violations_per_batch.append(batch_results["sla_violations"])
            safeguarded_services_per_batch.append(batch_results["safeguarded_services"])
            vulnerable_services_per_batch.append(batch_results["vulnerable_services"])

            if verbosity:
                print(f"    Maintenance Batch {batch_results['batch']} (duration={batch_results['batch_duration']}):")
                print(f"        Maintenance Duration: {batch_results['overall_maintenance_duration']}")
                print(f"        Overloaded Servers: {batch_results['overloaded_servers']}")
                print(f"        Used Servers: {batch_results['used_servers']}")
                print(f"        Updated Servers: {batch_results['updated_servers']}")
                print(f"        Outdated Servers: {batch_results['outdated_servers']}")
                print(f"        Vulnerability Surface: {batch_results['vulnerability_surface']}")
                print(f"        SLA Violations: {batch_results['sla_violations']}")
                print(f"        Safeguarded Services: {batch_results['safeguarded_services']}")
                print(f"        Vulnerable Services: {batch_results['vulnerable_services']}")
                print(f"        Number of Migrations: {batch_results['migrations']}")
                print(f"        Overall Migration_duration: {batch_results['overall_migration_duration']}")
                print(f"        Average Migration_duration: {batch_results['average_migration_duration']}")
                print(f"        Longest Migration_duration: {batch_results['longest_migration_duration']}")

        consolidation_rate = sum(consolidation_rate) / len(consolidation_rate)
        average_migration_duration = sum(average_migration_duration) / len(average_migration_duration)

        overall_results = [
            len(results),
            int(results[-1]["overall_maintenance_duration"]),
            consolidation_rate,
            sla_violations,
            number_of_migrations,
            overall_migration_duration,
            average_migration_duration,
            longest_migration_duration,
            vulnerability_surface,
            overloaded_servers,
            sla_violations_per_batch,
            updated_servers_per_batch,
            outdated_servers_per_batch,
            safeguarded_services_per_batch,
            vulnerable_services_per_batch,
            maintenance_duration_per_batch,
        ]

        print("    Overall:")
        print(f"        Maintenance Batches: {overall_results[0]}")
        print(f"        Maintenance Duration: {overall_results[1]}")
        print(f"        Consolidation Rate: {overall_results[2]}")
        print(f"        SLA Violations: {overall_results[3]}")
        print(f"        Migrations: {overall_results[4]}")
        print(f"        Overall Migration Duration: {overall_results[5]}")
        print(f"        Average Migration Duration: {overall_results[6]}")
        print(f"        Longest Migration Duration: {overall_results[7]}")
        print(f"        Vulnerability Surface: {overall_results[8]}")
        print(f"        Overloaded Servers: {overall_results[9]}")
        print(f"        SLA Violations per Batch: {overall_results[10]}")
        print(f"        Updated Servers per Batch: {overall_results[11]}")
        print(f"        Outdated Servers per Batch: {overall_results[12]}")
        print(f"        Safeguarded Services per Batch: {overall_results[13]}")
        print(f"        Vulnerable Services per Batch: {overall_results[14]}")
        print(f"        Maintenance Duration per Batch: {overall_results[15]}")
        print(f"        Migrations duration per batch: {migrations_duration_per_batch}")
        print(f"        All migrations: {migrations}")

        print(f"        CSV-READY RESULTS: ", end="")
        for metric in overall_results:
            print(f"{metric}", end="\t")
        print("")
