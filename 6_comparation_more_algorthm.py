import numpy as np
import pandas as pd
import random
import time

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

    # return global_best_position
    return list(map(int, global_best_position))

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

    # return best_solution
    return list(map(int, best_solution))

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
# Simple Neural Network (NN)
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
#Custom Decision Tree Regressor (DT)
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
#Custom Random Forest Regressor (RF)
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

def compare_algorithms(data, n_sensors):
    results = {}
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
    
    start_time = time.time()
    ge_positions = genetic_algorithm(data, n_sensors)
    ge_time = time.time() - start_time

    start_time = time.time()
    nn_positions = train_neural_network(data, n_sensors)
    nn_time = time.time() - start_time

    start_time = time.time()
    dt_positions = decision_tree_regressor(data, n_sensors)
    dt_time = time.time() - start_time

    start_time = time.time()
    fr_positions = random_forest_regressor(data, n_sensors)
    fr_time = time.time() - start_time

    # Fitness evaluation
    greedy_fitness = fitness_function(greedy_positions, step_data)
    pso_fitness = fitness_function(pso_positions, step_data)
    sa_fitness = fitness_function(sa_positions, step_data)
    ge_fitness = fitness_function(ge_positions, step_data)
    nn_fitness = fitness_function(nn_positions, step_data)
    dt_fitness = fitness_function(dt_positions, step_data)
    fr_fitness = fitness_function(fr_positions, step_data)

    results['greedy'] = f"Greedy Algorithm: Positions={greedy_positions}, Fitness={-greedy_fitness}, Time={greedy_time:.2f} seconds"
    results['pso'] = f"PSO Algorithm: Positions={pso_positions}, Fitness={-pso_fitness}, Time={pso_time:.2f} seconds"
    results['sa'] = f"SA Algorithm: Positions={sa_positions}, Fitness={-sa_fitness}, Time={sa_time:.2f} seconds"
    results['ge'] = f"Genetic Algorithm: Positions={ge_positions}, Fitness={-ge_fitness}, Time={ge_time:.2f} seconds"
    results['nn'] = f"Neural Network Algorithm: Positions={nn_positions}, Fitness={-nn_fitness}, Time={nn_time:.2f} seconds"
    results['dt'] = f"Decision Tree Algorithm: Positions={dt_positions}, Fitness={-dt_fitness}, Time={dt_time:.2f} seconds"
    results['fr'] = f"Random Forest Algorithm: Positions={fr_positions}, Fitness={-fr_fitness}, Time={fr_time:.2f} seconds"

    return results

# Number of sensors to place
n_sensors = 2


results = compare_algorithms(data, n_sensors)
for algorithm, result in results.items():
    print(result)
