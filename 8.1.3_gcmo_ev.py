import pandas as pd
import ast
import csv
import numpy as np
import networkx as nx
from epyt import epanet


def recompute_eigenvector_centrality(N,max_iter=500):
    netwx = nx.DiGraph()
    for keyLink,valueLink in enumerate(N.getLinkIndex()):
        link=N.getNodesConnectingLinksID(valueLink)[0]
        netwx.add_edge(link[0], link[1],name=valueLink)
    centrality = nx.eigenvector_centrality(netwx, max_iter=max_iter)
    # edge_labels = nx.get_edge_attributes(netwx, 'name')
    # print({val:(int(a),int(b)) for (a,b),val in edge_labels.items()})
    return centrality

def recompute_degree_centrality(N):
    netwx = nx.DiGraph()
    for keyLink,valueLink in enumerate(N.getLinkIndex()):
        link=N.getNodesConnectingLinksID(valueLink)[0]
        netwx.add_edge(link[0], link[1],name=valueLink)
    centrality = nx.degree_centrality(netwx)
    # edge_labels = nx.get_edge_attributes(netwx, 'name')
    # print({val:(int(a),int(b)) for (a,b),val in edge_labels.items()})
    return centrality

# Calculate coverage
def calculate_coverage(allnode,sensor_positions, step_data,steps,label):
    all_values = []
    step_data_min={column:[step_data[column][label-1]] for column in step_data.keys()}
    for key, values in step_data_min.items():
        for val in values:
            all_values.extend(val[1:-1].split(","))
    all_values = [int(x.strip()) for x in all_values if x!='']
    unique_values = list(set(all_values))
    # data_copy = resultLink.copy()
    # data_copy['Frequency'] = data_copy.iloc[:, 1:].sum(axis=1)
    # if label==2:
    #     print(label)
    #     print(data_copy['Frequency'])
    #     print(unique_values)
    #     exit()
    # n_nodes = len(step_data[steps[0]])
    n_nodes = len(unique_values)
    total_detected = np.zeros(n_nodes, dtype=int)
    for step in steps:
        detected = np.zeros(n_nodes, dtype=int)
        for sensor in sensor_positions:
            detected |= [sensor in np.fromstring(step_data_min[step][0][1:-1], dtype=np.int8, sep=', ') for s in unique_values]
        total_detected |= detected
    coverage = np.sum(total_detected) / len(allnode) * 100  
    # coverage = np.sum(total_detected) / n_nodes * 100  
    # coverage = np.sum(total_detected)
    # if label==2:
    #     # print(total_detected)
    #     # print(sensor_positions)
    #     print(coverage)
    #     exit()
    # exit()
    return coverage
def centrality_guided_optimization(n_sensors,step_data,steps,label,eg=None):
    all_values = []
    step_data_min={column:[step_data[column][label-1]] for column in step_data.keys()}
    
    for key, values in step_data_min.items():
        for val in values:
            all_values.extend(val[1:-1].split(","))
    all_values = [int(x.strip()) for x in all_values if x!='']
    unique_values = list(set(all_values))
    # print(step_data_min)
    # exit()
    G = nx.Graph()
    for step in steps:
        for i, nodes in enumerate(step_data_min[step]):
            nodes=np.fromstring(nodes[1:-1], dtype=np.int8, sep=', ')
            for node in nodes:
                G.add_edge(i+1, node)  # Shift node indices by 1 to avoid node 0
    # Calculate centrality measures
    degree_centrality = nx.degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    # print(unique_values)
    # print(degree_centrality)
    # exit()
    # Combine centrality measures into a score
    if eg!=None:
        egcentrality = nx.eigenvector_centrality(G, max_iter=2000)
        combined_centrality = {n: (degree_centrality[n] + closeness_centrality[n] + betweenness_centrality[n])* egcentrality[n] for n in unique_values}
    else:
        combined_centrality = {n: degree_centrality[n] + closeness_centrality[n] + betweenness_centrality[n] for n in unique_values}
    
    # Select top n_sensors based on combined centrality
    sorted_nodes = sorted(combined_centrality, key=combined_centrality.get, reverse=True)
    # print('===')
    sensor_placement=sorted_nodes[:n_sensors]
    frequency=[combined_centrality[k] for k in combined_centrality if k in sensor_placement]
    # frequency=dict((k, combined_centrality[k]) for k in combined_centrality if k in sensor_placement)
    return sensor_placement,frequency

def cgm_optimization(data, num_sensors, centrality, step_data, steps,label):
    """
    Centrality Guided Multi-Objective Optimization untuk penempatan sensor.

    Args:
    data: DataFrame yang berisi sumber kontaminan dan pergerakan kontaminan setiap interval waktu.
    num_sensors: Jumlah sensor yang akan ditempatkan.
    centrality: Dictionary dengan node sebagai key dan eigenvector centrality sebagai value.
    step_data: Data pergerakan kontaminan pada setiap interval waktu.
    steps: Daftar langkah waktu.

    Returns:
    sensor_locations: Daftar lokasi sensor terbaik berdasarkan optimasi multi-objektif.
    """
    # Inisialisasi daftar lokasi sensor
    sensor_locations = []
    score_set=[]
    # Skor awal node berdasarkan centrality dan cakupan deteksi
    
    # scores = {node: node for node in data['Source'].keys()}
    scores = {link:link for link in data['Source']}
    step_data_min={column:[step_data[column][label-1]] for column in step_data.keys()}

    # Menjalankan optimasi multi-objektif
    for _ in range(num_sensors):
        best_node = None
        best_score = -1
        best_coverage = None
        for node in scores.keys():
            # Cakupan deteksi jika sensor ditempatkan di node ini
            coverage = calculate_coverage(sensor_locations + [node], step_data_min, steps)
            # Kombinasikan skor centrality dan cakupan deteksi
            # score = centrality[node] * coverage
            score = centrality[label] * coverage
            # print(node,centrality[label],coverage,score)
            if score > best_score:
                best_score = score
                best_node = node
                best_coverage = round(best_score,4)

        if best_node is not None:
            sensor_locations.append(int(best_node))
            score_set.append(best_coverage)
            del scores[best_node]
    # if label==1:
    #     print(sensor_locations)
    #     print(score_set)
    # exit()
    return sensor_locations,list(map(float,score_set))

#load csv hasil generate time_contamination
# pathcsv='source_inp/time_contamination'
directory='source_inp/output_simulation'
pathcsv=f'{directory}/time_contamination'
networkset=['fos']#,'bwsn'
listNetwork=['FOS - unvertices.inp']#,'BWSN-clean.inp'
for i,ns in enumerate(networkset):
    N= epanet(f'source_inp/data_network/{listNetwork[i]}')
    N.plot_close()
    csvfile = open(f"{directory}/{ns}_8.1.3_gcmo_ev.csv", "w",newline='')
    writertank = csv.writer(csvfile, dialect='excel',delimiter=";")
    header=['Contaminant(Node)',
            'Sensor2(cgmo)',
            'score2(cgmo)',
            'Coverage2(cgmo)',
            'Sensor2(ev+cgmo)',
            'score2(ev+cgmo)',
            'Coverage2(ev+cgmo)',
            
            'Sensor3(cgmo)',
            'score3(cgmo)',
            'Coverage3(cgmo)',
            'Sensor3(ev+cgmo)',
            'score3(ev+cgmo)',
            'Coverage3(ev+cgmo)',

            'Sensor4(cgmo)',
            'score4(cgmo)',
            'Coverage4(cgmo)',
            'Sensor4(ev+cgmo)',
            'score4(ev+cgmo)',
            'Coverage4(ev+cgmo)',

            'Sensor5(cgmo)',
            'score5(cgmo)',
            'Coverage5(cgmo)',
            'Sensor5(ev+cgmo)',
            'score5(ev+cgmo)',
            'Coverage5(ev+cgmo)',
            ]
    readLink = pd.read_csv(f'{pathcsv}/{ns}_link.csv',on_bad_lines='skip',delimiter=';')

    # nodeSource=[i for i in range(1,len(readLink['NodeID'])+1)]
    nodeSource=[i for i in N.getLinkIndex()]

    writertank.writerow(header)

    steps = [col for col in readLink.columns if col.startswith('links_step')]
    step_data = {col: np.array([x if pd.isna(x)!=True else 0 for x in readLink[col]]) for col in steps}
    
    # centrality = recompute_eigenvector_centrality(N, max_iter=2000)
    # print(centrality)
    # print(len(centrality))
    # exit()
    # degre_centrality = recompute_degree_centrality(N)
    # centrality={int(key):val for key,val in centrality.items()}
    # degre_centrality={int(key):val for key,val in degre_centrality.items()}
    Allink = []
    step_data_min={column:[step_data[column][0]] for column in step_data.keys()}
    for key, values in step_data_min.items():
        for val in values:
            Allink.extend(val[1:-1].split(","))
    Allink = [int(x.strip()) for x in Allink if x!='']
    Allink = list(set(Allink))
    for id,label in enumerate(range(1,len(readLink['NodeID'])+1)):
        # dataResultLink={'Source':[]}
        # dataCopyLink=readLink.loc[[label-1]].copy()
        # for index,row in dataCopyLink.iterrows():
        #     for kr,ir in enumerate(nodeSource):
        #         dataResultLink['Source'].append(ir)
        #         node=ir
        #         for keys in dataCopyLink.keys():
        #             if keys!='NodeID' :
        #                 if keys not in dataResultLink:
        #                     dataResultLink[keys]=[]
        #                 val=0
        #                 if pd.isna(row[keys])!=True:
        #                     res=ast.literal_eval(row[keys])
        #                     if node in res:
        #                         val=1
        #                 dataResultLink[keys].append(val)
        # resultLink=pd.DataFrame(dataResultLink)
        
        # Hitung eigenvector centrality dengan iterasi maksimum lebih tinggi
        num_sensors = [2,3,4,5]
        sensor_list=[]
        # sensor_list=[centrality.get(str(label),0)]
        for sensor in num_sensors :
            sensor_locationsLinkOri,scoreLinkOri = centrality_guided_optimization(sensor,step_data, steps,label)
            # print(sensor_locationsLink,scoreLink)
            # exit()
            # sensor_locationsLink,scoreLink = cgm_optimization(resultLink, sensor, centrality, step_data, steps,label)
            sensor_locationsLink,scoreLink = centrality_guided_optimization(sensor,step_data, steps,label,eg=True)
            # sensor_locationsLinkDegre,scoreDegre = cgm_optimization(resultLink, sensor, degre_centrality, step_data, steps,label)
            coverage = calculate_coverage(Allink,sensor_locationsLinkOri,step_data,steps,label)
            coverage2 = calculate_coverage(Allink,sensor_locationsLink,step_data,steps,label)
            # print(coverage)
            # print(coverage2)
            # exit()
            sensor_list.append(sensor_locationsLinkOri)
            sensor_list.append(scoreLinkOri)
            sensor_list.append(f'{coverage:.2f}')
            sensor_list.append(sensor_locationsLink)
            sensor_list.append(scoreLink)
            sensor_list.append(f'{coverage2:.2f}')
        row=[label]+sensor_list
        writertank.writerow(row)
    csvfile.close()