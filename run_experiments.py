"""Contains a script that that executes a list of experiments automatically."""
# Python libraries
import itertools
import time
import os


def run_simulation(dataset: str, algorithm: str, n_gen: int, pop_size: int, cross_prob: float, weights: int) -> dict:
    """Executes the simulation with specified parameters."""
    # Running the simulation based on the parameters and gathering its execution time
    cmd = f"python3 -B -m simulator --dataset {dataset} --algorithm {algorithm} --n_gen {n_gen} --pop_size {pop_size} --cross_prob {cross_prob} --weights {weights}"
    print(f"    cmd = {cmd}")

    # Running the simulation with the specified parameters
    initial_time = time.time()

    stream = os.popen(cmd)
    output = stream.read()

    end_time = time.time()
    execution_time = round(end_time - initial_time, 7)

    # Parsing simulation results
    maintenance_batches = next(
        line.split("Maintenance Batches: ")[1] for line in output.splitlines() if "Maintenance Batches: " in line
    )
    maintenance_duration = next(
        line.split("Maintenance Duration: ")[1] for line in output.splitlines() if "Maintenance Duration: " in line
    )
    consolidation_rate = next(
        line.split("Consolidation Rate: ")[1] for line in output.splitlines() if "Consolidation Rate: " in line
    )
    sla_violations = next(
        line.split("SLA Violations: ")[1] for line in output.splitlines() if "SLA Violations: " in line
    )
    migrations = next(line.split("Migrations: ")[1] for line in output.splitlines() if "Migrations: " in line)
    overall_migration_duration = next(
        line.split("Overall Migration Duration: ")[1]
        for line in output.splitlines()
        if "Overall Migration Duration: " in line
    )
    avg_migration_duration = next(
        line.split("Average Migration Duration: ")[1]
        for line in output.splitlines()
        if "Average Migration Duration: " in line
    )
    longest_migration_duration = next(
        line.split("Longest Migration Duration: ")[1]
        for line in output.splitlines()
        if "Longest Migration Duration: " in line
    )
    vulnerability_surface = next(
        line.split("Vulnerability Surface: ")[1] for line in output.splitlines() if "Vulnerability Surface: " in line
    )
    overloaded_servers = next(
        line.split("Overloaded Servers: ")[1] for line in output.splitlines() if "Overloaded Servers: " in line
    )
    sla_violations_per_batch = next(
        line.split("SLA Violations per Batch: ")[1]
        for line in output.splitlines()
        if "SLA Violations per Batch: " in line
    )
    updated_servers_per_batch = next(
        line.split("Updated Servers per Batch: ")[1]
        for line in output.splitlines()
        if "Updated Servers per Batch: " in line
    )
    outdated_servers_per_batch = next(
        line.split("Outdated Servers per Batch: ")[1]
        for line in output.splitlines()
        if "Outdated Servers per Batch: " in line
    )
    safeguarded_services_per_batch = next(
        line.split("Safeguarded Services per Batch: ")[1]
        for line in output.splitlines()
        if "Safeguarded Services per Batch: " in line
    )
    vulnerable_services_per_batch = next(
        line.split("Vulnerable Services per Batch: ")[1]
        for line in output.splitlines()
        if "Vulnerable Services per Batch: " in line
    )
    maintenance_duration_per_batch = next(
        line.split("Maintenance Duration per Batch: ")[1]
        for line in output.splitlines()
        if "Maintenance Duration per Batch: " in line
    )
    migrations_duration_per_batch = next(
        line.split("Migrations duration per batch: ")[1]
        for line in output.splitlines()
        if "Migrations duration per batch: " in line
    )
    duration_of_each_migration = next(
        line.split("All migrations: ")[1] for line in output.splitlines() if "All migrations: " in line
    )

    result = {
        "dataset": dataset,
        "algorithm": algorithm,
        "n_gen": n_gen,
        "pop_size": pop_size,
        "cross_prob": cross_prob,
        "weights": weights,
        "execution_time": execution_time,
        "maintenance_batches": maintenance_batches,
        "maintenance_duration": maintenance_duration,
        "consolidation_rate": consolidation_rate,
        "sla_violations": sla_violations,
        "migrations": migrations,
        "overall_migration_duration": overall_migration_duration,
        "avg_migration_duration": avg_migration_duration,
        "longest_migration_duration": longest_migration_duration,
        "vulnerability_surface": vulnerability_surface,
        "overloaded_servers": overloaded_servers,
        "sla_violations_per_batch": sla_violations_per_batch,
        "updated_servers_per_batch": updated_servers_per_batch,
        "outdated_servers_per_batch": outdated_servers_per_batch,
        "safeguarded_services_per_batch": safeguarded_services_per_batch,
        "vulnerable_services_per_batch": vulnerable_services_per_batch,
        "maintenance_duration_per_batch": maintenance_duration_per_batch,
        "migrations_duration_per_batch": migrations_duration_per_batch,
        "duration_of_each_migration": duration_of_each_migration,
    }

    return result


# Parameters
datasets = ["dataset1"]
algorithms = ["laxus"]
population_sizes = [120]
number_of_generations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
crossover_probabilities = [0.25, 0.5, 0.75, 1]
weights = [0]


# Create CSV file with headers
output_file_name = f"results_{time.time()}.csv"

headers = [
    "dataset",
    "algorithm",
    "n_gen",
    "pop_size",
    "cross_prob",
    "weights",
    "execution_time",
    "maintenance_batches",
    "maintenance_duration",
    "consolidation_rate",
    "sla_violations",
    "migrations",
    "overall_migration_duration",
    "avg_migration_duration",
    "longest_migration_duration",
    "vulnerability_surface",
    "overloaded_servers",
    "sla_violations_per_batch",
    "updated_servers_per_batch",
    "outdated_servers_per_batch",
    "safeguarded_services_per_batch",
    "vulnerable_services_per_batch",
    "maintenance_duration_per_batch",
    "migrations_duration_per_batch",
    "duration_of_each_migration",
]
headers_str = ""
for header in headers:
    headers_str += f"{header},"

with open(output_file_name, "w") as text_file:
    print(headers_str, file=text_file)


# Generating list of combinations with the parameters specified
combinations = list(
    itertools.product(datasets, algorithms, population_sizes, number_of_generations, crossover_probabilities, weights)
)

# Executing simulations and collecting results
results = []

print(f"EXECUTING {len(combinations)} COMBINATIONS")

for i, parameters in enumerate(combinations, 1):
    # Parsing parameters
    dataset = parameters[0]
    algorithm = parameters[1]
    pop_size = parameters[2]
    n_gen = parameters[3]
    cross_prob = parameters[4]
    weights = parameters[5]

    print(f"[Execution {i}]")
    print(
        f"    [{algorithm}] dataset = {dataset}. pop_size = {pop_size}. n_gen = {n_gen}. cross_prob = {cross_prob}. weights = {weights}"
    )

    # Executing algorithm
    results = run_simulation(
        dataset=dataset,
        algorithm=algorithm,
        pop_size=pop_size,
        n_gen=n_gen,
        cross_prob=cross_prob,
        weights=weights,
    )

    print("    RESULTS:")

    execution_results = ""
    for key, value in results.items():
        print(f"        {key} = {value}")
        metric = str(value).replace(",", ";")
        execution_results += f"{metric},"

    # Exporting the execution results to the CSV file
    with open(output_file_name, "a") as text_file:
        print(execution_results, file=text_file)
    print(execution_results)
