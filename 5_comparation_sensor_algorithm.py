import numpy as np
import pandas as pd
import random
import time

# Load the CSV file with a semicolon delimiter
directory='source_inp/output_simulation'
file_path = f'{directory}/time_contamination/fos_node.csv'
data = pd.read_csv(file_path,on_bad_lines='skip',delimiter=';')

# Process the data
node_ids = data['NodeID'].values
steps = [col for col in data.columns if col.startswith('node_step')]
step_data = {col: np.array([set(eval(x)) for x in data[col]]) for col in steps}

# Define the fitness function
def fitness_function(sensor_positions, step_data):
    total_detection_time = 0
    n_nodes = len(step_data[steps[0]])

    for step in steps:
        detected = np.zeros(n_nodes, dtype=bool)

        for sensor in sensor_positions:
            detected |= np.array([sensor in step_data[step][i] for i in range(n_nodes)])

        detection_time = detected.sum()
        total_detection_time += detection_time

    return -total_detection_time  # Negative because we want to minimize the detection time

# Greedy Algorithm
def greedy_sensor_placement(data, n_sensors):
    node_importance = {node: 0 for node in data['NodeID']}
    for step in steps:
        for nodes in data[step].apply(eval):
            for node in nodes:
                node_importance[node] += 1

    sorted_nodes = sorted(node_importance, key=node_importance.get, reverse=False)
    return sorted_nodes[:n_sensors]

# Particle Swarm Optimization (PSO)
def particle_swarm_optimization(data, n_sensors, n_particles=20, n_iterations=50, w=0.5, c1=1.5, c2=1.5):
    n_nodes = len(data['NodeID'])
    # particle_positions = np.array([np.random.choice(n_nodes, n_sensors, replace=False) for _ in range(n_particles)])
    particle_positions = np.array([np.random.choice(np.arange(1,n_nodes), n_sensors, replace=False) for _ in range(n_particles)])
    particle_velocities = np.random.rand(n_particles, n_sensors)

    personal_best_positions = particle_positions.copy()
    personal_best_scores = np.array([fitness_function(p, step_data) for p in particle_positions])
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    # global_best_score = personal_best_scores.min()
    global_best_score = personal_best_scores.max()
    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(n_sensors), np.random.rand(n_sensors)
            cognitive_velocity = c1 * r1 * (personal_best_positions[i] - particle_positions[i])
            social_velocity = c2 * r2 * (global_best_position - particle_positions[i])
            particle_velocities[i] = w * particle_velocities[i] + cognitive_velocity + social_velocity

            particle_positions[i] = (particle_positions[i] + particle_velocities[i]).astype(int)
            # particle_positions[i] = np.clip(particle_positions[i], 0, n_nodes - 1)
            particle_positions[i] = np.clip(particle_positions[i], 1, n_nodes)

            unique_positions = np.unique(particle_positions[i])
            if len(unique_positions) < n_sensors:
                additional_nodes = np.setdiff1d(np.arange(1,n_nodes), unique_positions)
                np.random.shuffle(additional_nodes)
                unique_positions = np.append(unique_positions, additional_nodes[:n_sensors - len(unique_positions)])
            particle_positions[i] = unique_positions[:n_sensors]

            score = fitness_function(particle_positions[i], step_data)
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particle_positions[i]
            if score > global_best_score:
                global_best_score = score
                global_best_position = particle_positions[i]

    return global_best_position

# Simulated Annealing (SA)
def simulated_annealing(data, initial_temp, final_temp, alpha, max_iterations, n_sensors):
    n_nodes = len(data['NodeID'])
    # current_solution = np.random.choice(n_nodes, n_sensors, replace=False).tolist()
    current_solution = np.random.choice(np.arange(1,n_nodes), n_sensors, replace=False).tolist()
    current_fitness = fitness_function(current_solution, step_data)
    best_solution = current_solution
    best_fitness = current_fitness
    temp = initial_temp

    for _ in range(max_iterations):
        if temp <= final_temp:
            break

        new_solution = current_solution.copy()
        index_to_change = random.randint(0, len(new_solution) - 1)
        # new_solution[index_to_change] = random.randint(0, n_nodes - 1)
        new_solution[index_to_change] = random.randint(1, n_nodes)
        # print(new_solution)
        # print(index_to_change)
        # exit()
        new_solution = list(np.unique(new_solution))
        additional_nodes = random.sample(list(set(range(n_nodes)) - set(new_solution)), n_sensors - len(new_solution))
        new_solution = new_solution + additional_nodes

        new_fitness = fitness_function(new_solution, step_data)

        if new_fitness > current_fitness or random.uniform(0, 1) < np.exp((new_fitness - current_fitness) / temp):
            current_solution = new_solution
            current_fitness = new_fitness

        if new_fitness > best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness

        temp *= alpha

    return best_solution



# Number of sensors to place
n_sensors = 2

# Run and time each algorithm
start_time = time.time()
greedy_positions = greedy_sensor_placement(data, n_sensors)
greedy_time = time.time() - start_time

start_time = time.time()
pso_positions = particle_swarm_optimization(data, n_sensors)
pso_time = time.time() - start_time

start_time = time.time()
sa_positions = simulated_annealing(data, 1000, 1, 0.9, 1000, n_sensors)
sa_time = time.time() - start_time

# Fitness evaluation
greedy_fitness = fitness_function(greedy_positions, step_data)
pso_fitness = fitness_function(pso_positions, step_data)
sa_fitness = fitness_function(sa_positions, step_data)

# Display results
print(f"Greedy Algorithm: Positions={greedy_positions}, Fitness={-greedy_fitness}, Time={greedy_time:.2f} seconds")
print(f"PSO Algorithm: Positions={pso_positions}, Fitness={-pso_fitness}, Time={pso_time:.2f} seconds")
print(f"SA Algorithm: Positions={sa_positions}, Fitness={-sa_fitness}, Time={sa_time:.2f} seconds")
