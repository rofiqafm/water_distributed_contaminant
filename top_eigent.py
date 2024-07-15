import pandas as pd
import ast
import networkx as nx

# Load the data
# file_path = 'path_to_your_file/bwsn_node.csv'
file_path = f'source_inp/output_simulation/time_contamination/bwsn_node.csv'
data = pd.read_csv(file_path, delimiter=';')

# Transform data from string to list of integers
for col in data.columns[1:]:
    data[col] = data[col].apply(lambda x: ast.literal_eval(x))

# Create a graph
G = nx.Graph()

# Add edges to the graph based on contamination detection data
for index, row in data.iterrows():
    node = row['NodeID']
    for col in data.columns[1:]:
        for connected_node in row[col]:
            G.add_edge(node, connected_node)

# Calculate eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(G)

# Convert to DataFrame for better visualization
eigenvector_centrality_df = pd.DataFrame.from_dict(eigenvector_centrality, orient='index', columns=['Eigenvector Centrality'])

# Get the top 5 highest eigenvector centrality values
top_5_eigenvector_centrality = eigenvector_centrality_df.nlargest(5, 'Eigenvector Centrality')
print(eigenvector_centrality)
exit()
print(eigenvector_centrality_df)
print(top_5_eigenvector_centrality)
