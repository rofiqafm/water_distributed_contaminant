import pandas as pd
import numpy as np
import time
import networkx as nx

# Greedy Algorithm
def greedy_sensor_placement(data, n_sensors,steps):
    n_nodes = len(data['NodeID'])
    node_importance = {int(node): 0 for node in np.arange(1, n_nodes+1)}

    for step in steps:
        for nodes in step_data[step]:
            for node in nodes:
                if node in node_importance:
                    node_importance[node] += 1
    sorted_nodes = sorted(node_importance, key=node_importance.get, reverse=False)
    sensor_placement=sorted_nodes[:n_sensors]
    frequency=dict((k, node_importance[k]) for k in node_importance if k in sensor_placement)
    return frequency,sensor_placement
    # return dict(sorted(node_importance.items(), key=lambda item: item[1],reverse=True)),sorted_nodes[:n_sensors]
# Centrality Guided Multi-Objective Optimization (CGMO)
def centrality_guided_optimization(data, n_sensors,steps):
    # Create a graph based on contamination detection steps
    G = nx.Graph()
    for step in steps:
        for i, nodes in enumerate(step_data[step]):
            for node in nodes:
                G.add_edge(i+1, node)  # Shift node indices by 1 to avoid node 0

    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Combine centrality measures into a score
    combined_centrality = {node: degree_centrality[node] + closeness_centrality[node] + betweenness_centrality[node] for node in G.nodes}
    print(betweenness_centrality)
    # Select top n_sensors based on combined centrality
    sorted_nodes = sorted(combined_centrality, key=combined_centrality.get, reverse=True)
    sensor_placement=sorted_nodes[:n_sensors]
    frequency=dict((k, combined_centrality[k]) for k in combined_centrality if k in sensor_placement)
    return frequency,sensor_placement

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

#proses menjalankan semua perbandingan setiap nilai pada algoritma yang di gunakan
def compare_algorithms(data, n_sensors,step_data,steps):
    results = {}
    # Menjalankan Semua perhitungan Algoritma
    #===============Greedy=======================
    start_time = time.time()
    greedy_frequency,greedy_positions = greedy_sensor_placement(data, n_sensors,steps)
    greedy_time = time.time() - start_time
    #===============CGMO=======================
    start_time = time.time()
    cgmo_frequency,cgmo_positions = centrality_guided_optimization(data, n_sensors,steps)
    cgmo_time = time.time() - start_time
    #===========================================
    #menjalankan semua fitness setiap algoritma
    # # Fitness evaluation
    greedy_fitness = fitness_function(greedy_positions, step_data)
    cgmo_fitness = fitness_function(cgmo_positions, step_data)
    #===========================================
    # Mendapatkan besaran coverage hasil penempatan sensor setiap algortima
    greedy_coverage=calculate_coverage(greedy_positions, step_data)
    cgmo_coverage = calculate_coverage(cgmo_positions, step_data)
    #===========================================
    # Mendapatkan prosentase precision dan recall berdasarkan penempatan sensor setiap algortima
    greedy_precision, greedy_recall=calculate_precision_recall(greedy_positions, step_data)
    cgmo_precision, cgmo_recall = calculate_precision_recall(cgmo_positions, step_data)

    #===========================================
    # output dalam bentuk List
    results=[
        f"Greedy Algorithm: Positions={greedy_positions}, Fitness={-greedy_fitness}, Coverage={greedy_coverage:.2f}%, Precision={greedy_precision}, Recall={greedy_recall}, Time={greedy_time:.2f} seconds",
        greedy_frequency,
        f"CGMO Algorithm: Positions={cgmo_positions}, Fitness={-cgmo_fitness}, Coverage={cgmo_coverage:.2f}%, Precision={cgmo_precision}, Recall={cgmo_recall}, Time={cgmo_time:.2f} seconds",
        cgmo_frequency
    ]
    return results

#julmlah Sensor yang di gunakan
n_sensors = [2,3,4,5]
name_network=['fos','bwsn']
# Load the CSV file with a semicolon delimiter
directory='source_inp/output_simulation'
f = open(f"{directory}/greedy_gcmo.txt", "w",newline='')
h=["Source","Frequency"]
for name in name_network:
    file_path = f'{directory}/time_contamination/{name}_node.csv'
    data = pd.read_csv(file_path,on_bad_lines='skip',delimiter=';')
    steps = [col for col in data.columns if col.startswith('node_step')]
    step_data = {col: np.array([set(eval(x)) for x in data[col]]) for col in steps}
    results = compare_algorithms(data, 5,step_data,steps)
    f.write(f"========{name.upper()}=======\n")
    for result in results :
        if isinstance(result, dict):    
            f.write(f"-----------------------\n")
            f.write(f"{h}\n")
            f.write(f"-----------------------\n")
            for key,val in result.items():
                f.write(f"{[key,val]}\n")
            f.write(f"====================\n")
        else:
            f.write(f"{result}\n")
    f.write(f"\n")
f.close()

