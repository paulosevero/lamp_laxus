"""Contains a set of methods that extend EdgeSimPy's dataset generator to create objects with maintenance-related attributes.
"""


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
