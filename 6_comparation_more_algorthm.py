import numpy as np
import pandas as pd
import random
import time
import networkx as nx

# Custom Neural Network (NN)
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.random.randn(output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        output = self.sigmoid(self.z2)
        return output

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        a1_error = output_delta.dot(self.W2.T)
        a1_delta = a1_error * self.sigmoid_derivative(self.a1)

        self.W2 += self.a1.T.dot(output_delta)
        self.b2 += np.sum(output_delta, axis=0)
        self.W1 += X.T.dot(a1_delta)
        self.b1 += np.sum(a1_delta, axis=0)

    def train(self, X, y, iterations=1000):
        for _ in range(iterations):
            output = self.forward(X)
            self.backward(X, y, output)
#====================================================
# Custom Decision Tree Regressor (DT)
class SimpleDecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth or len(y) < 2:
            return np.mean(y)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.mean(y)

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_mse = float('inf')

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_mse = np.var(y[left_indices]) * len(y[left_indices])
                right_mse = np.var(y[right_indices]) * len(y[right_indices])
                mse = (left_mse + right_mse) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] <= threshold:
            return self._predict_single(x, left_subtree)
        else:
            return self._predict_single(x, right_subtree)
#====================================================
# Custom Random Forest Regressor (RF)
class SimpleDecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth or len(y) < 2:
            return np.mean(y)

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            return np.mean(y)

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_mse = float('inf')

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                left_mse = np.var(y[left_indices]) * len(y[left_indices])
                right_mse = np.var(y[right_indices]) * len(y[right_indices])
                mse = (left_mse + right_mse) / len(y)

                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] <= threshold:
            return self._predict_single(x, left_subtree)
        else:
            return self._predict_single(x, right_subtree)
#====================================================
class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = [SimpleDecisionTree(max_depth=max_depth) for _ in range(n_trees)]

    def fit(self, X, y):
        for tree in self.trees:
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)
#====================================================
# Ant Colony Optimization (ACO)
class AntColonyOptimizer:
    def __init__(self, data, n_sensors, n_ants=20, n_best=5, n_iterations=100, decay=0.5, alpha=1, beta=2):
        self.data = data
        self.n_sensors = n_sensors
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.n_nodes = len(data['NodeID'])
        self.pheromone = np.ones((self.n_nodes, self.n_nodes)) / self.n_nodes

    def run(self):
        best_solution = None
        best_fitness = float('-inf')

        for _ in range(self.n_iterations):
            all_solutions = self.construct_solutions()
            self.spread_pheromone(all_solutions, self.n_best)
            best_current = max(all_solutions, key=lambda x: x[1])
            if best_current[1] > best_fitness:
                best_solution = best_current[0]
                best_fitness = best_current[1]
            self.pheromone *= self.decay

        return list(map(int, best_solution))

    def construct_solutions(self):
        all_solutions = []
        for _ in range(self.n_ants):
            solution = self.construct_solution()
            fitness = fitness_function(solution, step_data)
            all_solutions.append((solution, fitness))
        return all_solutions

    def construct_solution(self):
        solution = []
        available_nodes = set(range(1, self.n_nodes))  # Start from 1 to avoid node 0
        for _ in range(self.n_sensors):
            probabilities = self.calculate_probabilities(solution, available_nodes)
            next_node = self.select_next_node(probabilities, available_nodes)
            solution.append(next_node)
            available_nodes.remove(next_node)
        return solution

    def calculate_probabilities(self, solution, available_nodes):
        pheromone = self.pheromone
        probabilities = np.zeros(self.n_nodes)
        for node in available_nodes:
            probabilities[node] = (pheromone[node, node] ** self.alpha) * ((1.0 / (1 + np.sum(pheromone[node]))) ** self.beta)
        probabilities = np.maximum(probabilities, 0)  # Ensure non-negative
        total = np.sum(probabilities)
        if total == 0:
            probabilities[list(available_nodes)] = 1 / len(available_nodes)  # Equal probability if all are zero
        else:
            probabilities /= total  # Normalize to sum to 1
        return probabilities

    def select_next_node(self, probabilities, available_nodes):
        return np.random.choice(list(available_nodes), p=probabilities[list(available_nodes)])

    def spread_pheromone(self, all_solutions, n_best):
        sorted_solutions = sorted(all_solutions, key=lambda x: x[1], reverse=True)
        for solution, fitness in sorted_solutions[:n_best]:
            for node in solution:
                self.pheromone[node, node] += fitness
#====================================================
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
#====================================================
# Greedy Algorithm
def greedy_sensor_placement(data, n_sensors):
    node_importance = {node: 0 for node in data['NodeID']}
    for step in steps:
        for nodes in data[step].apply(eval):
            for node in nodes:
                node_importance[node] += 1

    sorted_nodes = sorted(node_importance, key=node_importance.get, reverse=False)
    return sorted_nodes[:n_sensors]
#====================================================
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

    # return global_best_position
    return list(map(int, global_best_position))
#====================================================
# Simulated Annealing (SA)
def simulated_annealing(data, initial_temp, final_temp, alpha, max_iterations, n_sensors):
    n_nodes = len(data['NodeID'])
    current_solution = np.random.choice(n_nodes, n_sensors, replace=False).tolist()
    # current_solution = np.random.choice(np.arange(1,n_nodes), n_sensors, replace=False).tolist()
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
        new_solution = list(np.unique(new_solution))
        # additional_nodes = random.sample(list(set(range(n_nodes)) - set(new_solution)), n_sensors - len(new_solution))
        additional_nodes = random.sample(list(set(range(1,n_nodes)) - set(new_solution)), n_sensors - len(new_solution))
        new_solution = new_solution + additional_nodes

        new_fitness = fitness_function(new_solution, step_data)

        if new_fitness > current_fitness or random.uniform(0, 1) < np.exp((new_fitness - current_fitness) / temp):
            current_solution = new_solution
            current_fitness = new_fitness

        if new_fitness > best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness

        temp *= alpha

    # return best_solution
    return list(map(int, best_solution))
#====================================================
# Genetic Algorithm (GA)
def genetic_algorithm(data, n_sensors, pop_size=20, n_generations=50, mutation_rate=0.1):
    n_nodes = len(data['NodeID'])
    
    def create_individual():
        return np.random.choice(np.arange(1, n_nodes), n_sensors, replace=False)

    def crossover(parent1, parent2):
        # crossover_point = np.random.randint(1, n_sensors - 1)
        crossover_point = np.random.randint(1, n_sensors)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(individual):
        if np.random.rand() < mutation_rate:
            index_to_change = np.random.randint(n_sensors)
            new_value = np.random.randint(1, n_nodes)
            individual[index_to_change] = new_value
        return individual

    population = np.array([create_individual() for _ in range(pop_size)])
    fitness_scores = np.array([fitness_function(individual, step_data) for individual in population])

    for _ in range(n_generations):
        new_population = []
        fitness_scores = np.array([fitness_function(individual, step_data) for individual in population])
        sorted_indices = np.argsort(fitness_scores)
        population = population[sorted_indices]
        fitness_scores = fitness_scores[sorted_indices]

        new_population.extend(population[:pop_size//2])
        
        for _ in range(pop_size // 2):
            parents = np.random.choice(pop_size // 2, 2, replace=False)
            child1, child2 = crossover(population[parents[0]], population[parents[1]])
            new_population.extend([mutate(child1), mutate(child2)])

        population = np.array(new_population)
        fitness_scores = np.array([fitness_function(individual, step_data) for individual in population])

    best_individual = population[np.argmax(fitness_scores)]
    # return best_individual
    return list(map(int, best_individual))
#====================================================
# # Simple Neural Network (NN)
def train_neural_network(data, n_sensors):
    n_nodes = len(data['NodeID'])
    input_size = n_sensors
    hidden_size = 10
    output_size = 1

    # Generate synthetic data for training
    X_train = []
    y_train = []

    for _ in range(1000):  # Number of synthetic samples
        sensors = np.random.choice(np.arange(1, n_nodes), n_sensors, replace=False)
        fitness = fitness_function(sensors, step_data)
        X_train.append(sensors)
        y_train.append(fitness)

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1)

    # Train the neural network
    nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X_train, y_train, iterations=1000)

    # Predict the best sensor positions
    best_positions = nn.forward(X_train)
    best_sensor_positions = X_train[np.argmax(best_positions)]

    # return best_sensor_positions
    return list(map(int, best_sensor_positions))
#====================================================
# #Custom Decision Tree Regressor (DT)
def decision_tree_regressor(data, n_sensors):
    n_nodes = len(data['NodeID'])

    # Generate synthetic data for training
    X_train = []
    y_train = []

    for _ in range(1000):  # Number of synthetic samples
        sensors = np.random.choice(np.arange(1, n_nodes), n_sensors, replace=False)
        fitness = fitness_function(sensors, step_data)
        X_train.append(sensors)
        y_train.append(fitness)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train a decision tree regressor
    model = SimpleDecisionTree()
    model.fit(X_train, y_train)

    # Predict the best sensor positions
    best_positions = model.predict(X_train)
    best_sensor_positions = X_train[np.argmax(best_positions)]

    return list(map(int, best_sensor_positions))
#====================================================
# #Custom Random Forest Regressor (RF)
def random_forest_regressor(data, n_sensors):
    n_nodes = len(data['NodeID'])

    # Generate synthetic data for training
    X_train = []
    y_train = []

    for _ in range(1000):  # Number of synthetic samples
        sensors = np.random.choice(np.arange(1, n_nodes), n_sensors, replace=False)
        fitness = fitness_function(sensors, step_data)
        X_train.append(sensors)
        y_train.append(fitness)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Train a random forest regressor
    model = SimpleRandomForest(n_trees=10)
    model.fit(X_train, y_train)

    # Predict the best sensor positions
    best_positions = model.predict(X_train)
    best_sensor_positions = X_train[np.argmax(best_positions)]

    return list(map(int, best_sensor_positions))
#====================================================
# Custom K-means Clustering
def custom_kmeans(data, n_sensors, max_iterations=100):
    n_nodes = len(data['NodeID'])
    node_positions = np.arange(1, n_nodes)

    # Initialize centroids
    centroids = np.random.choice(node_positions, n_sensors, replace=False)

    for _ in range(max_iterations):
        clusters = {i: [] for i in centroids}
        
        # Assign nodes to the nearest centroid
        for node in node_positions:
            closest_centroid = min(centroids, key=lambda c: abs(c - node))
            clusters[closest_centroid].append(node)
        
        new_centroids = []
        for centroid in centroids:
            if clusters[centroid]:  # Avoid division by zero
                new_centroid = int(np.mean(clusters[centroid]))
                new_centroids.append(new_centroid)
            else:
                new_centroids.append(centroid)
        
        new_centroids = np.array(new_centroids)
        if np.array_equal(centroids, new_centroids):
            break
        
        centroids = new_centroids

    # return list(centroids)
    return list(map(int, centroids))
#====================================================
# Centrality Guided Multi-Objective Optimization (CGMO)
def centrality_guided_optimization(data, n_sensors):
    # Create a graph based on contamination detection steps
    G = nx.Graph()
    for step in steps:
        for i, nodes in enumerate(step_data[step]):
            for node in nodes:
                G.add_edge(i+1, node)  # mengeluarkan nilai key 0 dari daftar node, krn node di mulai dari 1 - total(n_sensors)

    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Combine centrality measures into a score
    combined_centrality = {node: degree_centrality[node] + closeness_centrality[node] + betweenness_centrality[node] for node in G.nodes}

    # Select top n_sensors based on combined centrality
    sorted_nodes = sorted(combined_centrality, key=combined_centrality.get, reverse=True)
    return sorted_nodes[:n_sensors]
#====================================================
# Calculate coverage
def calculate_coverage(sensor_positions, step_data):
    n_nodes = len(step_data[steps[0]])
    total_detected = np.zeros(n_nodes, dtype=bool)

    for step in steps:
        detected = np.zeros(n_nodes, dtype=bool)
        for sensor in sensor_positions:
            detected |= np.array([sensor in step_data[step][i] for i in range(n_nodes)])
        total_detected |= detected

    coverage = np.sum(total_detected) / n_nodes * 100  # Percentage of nodes covered
    return coverage
#====================================================
# Calculate precision and recall
def calculate_precision_recall(sensor_positions, step_data):
    n_nodes = len(step_data[steps[0]])
    total_detected = np.zeros(n_nodes, dtype=bool)
    actual_contaminated = np.zeros(n_nodes, dtype=bool)

    for step in steps:
        detected = np.zeros(n_nodes, dtype=bool)
        for sensor in sensor_positions:
            detected |= np.array([sensor in step_data[step][i] for i in range(n_nodes)])
        total_detected |= detected

        for nodes in step_data[step]:
            actual_contaminated |= np.array([True if i in nodes else False for i in range(n_nodes)])

    true_positives = np.sum(total_detected & actual_contaminated)
    false_positives = np.sum(total_detected & ~actual_contaminated)
    false_negatives = np.sum(~total_detected & actual_contaminated)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall
#====================================================

#proses menjalankan semua perbandingan setiap nilai pada algoritma yang di gunakan
def compare_algorithms(data, n_sensors):
    results = {}
    # Menjalankan Semua perhitungan Algoritma
    #===============Greedy=======================
    start_time = time.time()
    greedy_positions = greedy_sensor_placement(data, n_sensors)
    greedy_time = time.time() - start_time
    #===============PSO=======================
    start_time = time.time()
    pso_positions = particle_swarm_optimization(data, n_sensors)
    pso_time = time.time() - start_time
    #===============SA=======================
    start_time = time.time()
    sa_positions = simulated_annealing(data, 1000, 1, 0.9, 1000, n_sensors) # data, initial_temp, final_temp, alpha, max_iterations, n_sensors
    sa_time = time.time() - start_time
    #===============Genetik=======================
    start_time = time.time()
    ge_positions = genetic_algorithm(data, n_sensors)
    ge_time = time.time() - start_time
    #===============Neural-Network=======================
    start_time = time.time()
    nn_positions = train_neural_network(data, n_sensors)
    nn_time = time.time() - start_time
    #===============Decision-Tree=======================
    start_time = time.time()
    dt_positions = decision_tree_regressor(data, n_sensors)
    dt_time = time.time() - start_time
    #===============Random-Forest=======================
    start_time = time.time()
    fr_positions = random_forest_regressor(data, n_sensors)
    fr_time = time.time() - start_time
    #===============ACO=======================
    start_time = time.time()
    aco = AntColonyOptimizer(data, n_sensors)
    aco_positions = aco.run()
    aco_time = time.time() - start_time
    #===============KMeans=======================
    start_time = time.time()
    kmeans_positions = custom_kmeans(data, n_sensors)
    kmeans_time = time.time() - start_time
    #===============CGMO=======================
    start_time = time.time()
    cgmo_positions = centrality_guided_optimization(data, n_sensors)
    cgmo_time = time.time() - start_time
    #===========================================
    #menjalankan semua fitness setiap algoritma
    # # Fitness evaluation
    greedy_fitness = fitness_function(greedy_positions, step_data)
    pso_fitness = fitness_function(pso_positions, step_data)
    sa_fitness = fitness_function(sa_positions, step_data)
    ge_fitness = fitness_function(ge_positions, step_data)
    nn_fitness = fitness_function(nn_positions, step_data)
    dt_fitness = fitness_function(dt_positions, step_data)
    fr_fitness = fitness_function(fr_positions, step_data)
    aco_fitness = fitness_function(aco_positions, step_data)
    kmeans_fitness = fitness_function(kmeans_positions, step_data)
    cgmo_fitness = fitness_function(cgmo_positions, step_data)
    #===========================================
    # Mendapatkan besaran coverage hasil penempatan sensor setiap algortima
    greedy_coverage=calculate_coverage(greedy_positions, step_data)
    pso_coverage=calculate_coverage(pso_positions, step_data)
    sa_coverage=calculate_coverage(sa_positions, step_data)
    ge_coverage=calculate_coverage(ge_positions, step_data)
    nn_coverage=calculate_coverage(nn_positions, step_data)
    dt_coverage=calculate_coverage(dt_positions, step_data)
    fr_coverage=calculate_coverage(fr_positions, step_data)
    aco_coverage = calculate_coverage(aco_positions, step_data)
    kmeans_coverage = calculate_coverage(kmeans_positions, step_data)
    cgmo_coverage = calculate_coverage(cgmo_positions, step_data)
    #===========================================
    # Mendapatkan prosentase precision dan recall berdasarkan penempatan sensor setiap algortima
    greedy_precision, greedy_recall=calculate_precision_recall(greedy_positions, step_data)
    pso_precision, pso_recall=calculate_precision_recall(pso_positions, step_data)
    sa_precision, sa_recall=calculate_precision_recall(sa_positions, step_data)
    ge_precision, ge_recall=calculate_precision_recall(ge_positions, step_data)
    nn_precision, nn_recall=calculate_precision_recall(nn_positions, step_data)
    dt_precision, dt_recall=calculate_precision_recall(dt_positions, step_data)
    fr_precision, fr_recall=calculate_precision_recall(fr_positions, step_data)
    aco_precision, aco_recall = calculate_precision_recall(aco_positions, step_data)
    kmeans_precision, kmeans_recall = calculate_precision_recall(kmeans_positions, step_data)
    cgmo_precision, cgmo_recall = calculate_precision_recall(cgmo_positions, step_data)

    #===========================================
    # output dalam bentuk List
    results['greedy'] = f"Greedy Algorithm: Positions={greedy_positions}, Fitness={-greedy_fitness}, Coverage={greedy_coverage:.2f}%, Precision={greedy_precision}, Recall={greedy_recall}, Time={greedy_time:.2f} seconds"
    results['pso'] = f"PSO Algorithm: Positions={pso_positions}, Fitness={-pso_fitness}, Coverage={pso_coverage:.2f}%, Precision={pso_precision}, Recall={pso_recall}, Time={pso_time:.2f} seconds"
    results['sa'] = f"SA Algorithm: Positions={sa_positions}, Fitness={-sa_fitness}, Coverage={sa_coverage:.2f}%, Precision={sa_precision}, Recall={sa_recall}, Time={sa_time:.2f} seconds"
    results['ge'] = f"Genetic Algorithm: Positions={ge_positions}, Fitness={-ge_fitness}, Coverage={ge_coverage:.2f}%, Precision={ge_precision}, Recall={ge_recall}, Time={ge_time:.2f} seconds"
    results['nn'] = f"Neural Network Algorithm: Positions={nn_positions}, Fitness={-nn_fitness}, Coverage={nn_coverage:.2f}%, Precision={nn_precision}, Recall={nn_recall}, Time={nn_time:.2f} seconds"
    results['dt'] = f"Decision Tree Algorithm: Positions={dt_positions}, Fitness={-dt_fitness}, Coverage={dt_coverage:.2f}%, Precision={dt_precision}, Recall={dt_recall}, Time={dt_time:.2f} seconds"
    results['fr'] = f"Random Forest Algorithm: Positions={fr_positions}, Fitness={-fr_fitness}, Coverage={fr_coverage:.2f}%, Precision={fr_precision}, Recall={fr_recall}, Time={fr_time:.2f} seconds"
    results['aco'] = f"ACO Algorithm: Positions={aco_positions}, Fitness={-aco_fitness}, Coverage={aco_coverage:.2f}%, Precision={aco_precision}, Recall={aco_recall}, Time={aco_time:.2f} seconds"
    results['kmeans'] = f"K-means Clustering Algorithm: Positions={kmeans_positions}, Fitness={-kmeans_fitness}, Coverage={kmeans_coverage:.2f}%, Precision={kmeans_precision}, Recall={kmeans_recall}, Time={kmeans_time:.2f} seconds"
    results['cgmo'] = f"CGMO Algorithm: Positions={cgmo_positions}, Fitness={-cgmo_fitness}, Coverage={cgmo_coverage:.2f}%, Precision={cgmo_precision}, Recall={cgmo_recall}, Time={cgmo_time:.2f} seconds"

    return results
# 
#julmlah Sensor yang di gunakan
n_sensors = [2,3,4,5]
# Load the CSV file with a semicolon delimiter
directory='source_inp/output_simulation'

#===========================================Network FOS =====================================================
#membaca data network fos
file_path = f'{directory}/time_contamination/fos_node.csv'
data = pd.read_csv(file_path,on_bad_lines='skip',delimiter=';')

# Process inisialisasi step, dan step_data
node_ids = data['NodeID'].values
steps = [col for col in data.columns if col.startswith('node_step')]
step_data = {col: np.array([set(eval(x)) for x in data[col]]) for col in steps}
#write file
f = open(f"{directory}/fos_comparation_sensor_placement.txt", "w",newline='')
print(f"Network : FOS")
print(f"=================")
f.write(f"Network : FOS\n")
f.write(f"================\n")
#melakukan perulangan untuk menjalankan fungsi compare algorima dan menampilkan hasilnya setiap sebanyak jumlah sensor yang telah di tetapkan
for sensor in n_sensors :
    results = compare_algorithms(data, sensor)
    print(f"Jumlah Sensor : {sensor}")
    f.write(f"Jumlah Sensor : {sensor}\n")
    print(f"----------------------------")
    f.write(f"----------------------------\n")
    for algorithm, result in results.items():
        print(result)
        f.write(f'{result}\n')
    print(f"============================\n")
    f.write(f"============================\n")
f.close()
#===========================================Network Jilin =====================================================
#membaca data network Jilin
file_path = f'{directory}/time_contamination/jilin_node.csv'
data = pd.read_csv(file_path,on_bad_lines='skip',delimiter=';')

# Process the data
node_ids = data['NodeID'].values
steps = [col for col in data.columns if col.startswith('node_step')]
step_data = {col: np.array([set(eval(x)) for x in data[col]]) for col in steps}

#write file
f = open(f"{directory}/jilin_comparation_sensor_placement.txt", "w",newline='')
# Number of sensors to place
print(f"Network : Jilin")
print(f"=================")
f.write(f"Network : Jilin\n")
f.write(f"================\n")
#melakukan perulangan untuk menjalankan fungsi compare algorima dan menampilkan hasilnya setiap sebanyak jumlah sensor yang telah di tetapkan
for sensor in n_sensors :
    results = compare_algorithms(data, sensor)
    print(f"Jumlah Sensor : {sensor}")
    f.write(f"Jumlah Sensor : {sensor}\n")
    print(f"----------------------------")
    f.write(f"----------------------------\n")
    for algorithm, result in results.items():
        print(result)
        f.write(f'{result}\n')
    print(f"============================\n")
    f.write(f"============================\n")
f.close()
#===========================================Network BWSN =====================================================
#membaca data network BWSN
file_path = f'{directory}/time_contamination/bwsn_node.csv'
data = pd.read_csv(file_path,on_bad_lines='skip',delimiter=';')

# Process the data
node_ids = data['NodeID'].values
steps = [col for col in data.columns if col.startswith('node_step')]
step_data = {col: np.array([set(eval(x)) for x in data[col]]) for col in steps}

#write file
f = open(f"{directory}/bwsn_comparation_sensor_placement.txt", "w",newline='')
# Number of sensors to place
print(f"Network : bwsn")
print(f"=================")
f.write(f"Network : bwsn\n")
f.write(f"================\n")
#melakukan perulangan untuk menjalankan fungsi compare algorima dan menampilkan hasilnya setiap sebanyak jumlah sensor yang telah di tetapkan
for sensor in n_sensors :
    results = compare_algorithms(data, sensor)
    print(f"Jumlah Sensor : {sensor}")
    f.write(f"Jumlah Sensor : {sensor}\n")
    print(f"----------------------------")
    f.write(f"----------------------------\n")
    for algorithm, result in results.items():
        print(result)
        f.write(f'{result}\n')
    print(f"============================\n")
    f.write(f"============================\n")
f.close()


