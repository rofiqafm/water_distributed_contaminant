from epyt import epanet
import networkx as nx
import pandas as pd
import numpy as np
import ast
import csv

np.seterr(divide='ignore', invalid='ignore')

def readDataFrame(readLink,label,nodeSource,valDefault=0):
# =====================================
    #DataFrame untuk Frequency greedy original
    dataResultLink={'Source':[]}
    dataCopyLink=readLink.loc[[label-1]].copy()
    for index,row in dataCopyLink.iterrows():
        for kr,ir in enumerate(nodeSource):
            dataResultLink['Source'].append(ir)
            node=ir
            for keys in dataCopyLink.keys():
                if keys!='NodeID' :
                    if keys not in dataResultLink:
                        dataResultLink[keys]=[]
                    val=valDefault
                    if pd.isna(row[keys])!=True:
                        res=ast.literal_eval(row[keys])
                        if node in res:
                            val=1
                    dataResultLink[keys].append(val)
    resultLink=pd.DataFrame(dataResultLink)
    return resultLink

def combine_centrality_compute(N,step_data,label=None,normal=None):
    # =====================================
    # Menghitung Eigenvector Centrality
    ev,bw = compute_centrality(N,step_data,label)
    # Normalisasi nilai centrality
    eigen_values = np.array(list(ev.values()))
    between_values = np.array(list(bw.values()))
    if normal==True:
        # eigen_norm = (eigen_values - eigen_values.min()) / (eigen_values.max() - eigen_values.min())
        eigen_norm = eigen_values / eigen_values.max() 
        # between_norm = (between_values - between_values.min()) / (between_values.max() - between_values.min())
        between_norm = between_values / between_values.max() 
    else:
        eigen_norm=eigen_values
        between_norm=between_values
    # Kombinasi nilai centrality
    alpha = 0.5  # Bobot untuk kombinasi
    combined_centrality = alpha * eigen_norm + (1 - alpha) * between_norm
    # combined_centrality = eigen_values + between_values
    # Menampilkan nilai kombinasi centrality
    combined_centrality={i:a for i,a in enumerate(combined_centrality)}
    combined_dict = {l+1: combined_centrality[l] for l in combined_centrality}
    # sorted_nodes = sorted(combined_dict, key=combined_dict.get, reverse=True)
    # ====================================
    return combined_dict

def compute_centrality(netwx,step_data,label):
    all_values = []
    step_data_min={column:[step_data[column][label-1]] for column in step_data.keys()}
    # netwx = nx.Graph()
    for step in steps:
        for i, nodes in enumerate(step_data_min[step]):
            nodes=np.fromstring(nodes[1:-1], dtype=np.int8, sep=', ')
            for node in nodes:
                netwx.add_edge(i+1, node) 
    eigenvector_centrality = nx.eigenvector_centrality(netwx,max_iter=2000)
    betweenness_centrality = nx.betweenness_centrality(netwx)
    return eigenvector_centrality,betweenness_centrality#,unique_values
# Fungsi Greedy untuk Penempatan Sensor
def greedy_sensor_placement(graph, combined_centrality, num_sensors):
    sorted_nodes = sorted(combined_centrality, key=combined_centrality.get, reverse=True)
    sensor_placement = sorted_nodes[:num_sensors]
    score_centrality=[]
    for sp in sensor_placement:
        if pd.isna(combined_centrality[sp])!=True:
            nilai=combined_centrality[sp]
        else:
            nilai=0
        score_centrality.append(nilai)
    return sensor_placement,list(map(float, score_centrality))
def greedy_sensor_placement_original(data, num_sensors,label,normal=None):
    data_copy = data.copy()
    data_copy['Frequency'] = data_copy.iloc[:, 1:].sum(axis=1)
    sensor_locations = []
    frequency_locations = []
    for _ in range(num_sensors):
        max_index = data_copy['Frequency'].idxmax()
        sensor_locations.append(int(data_copy.at[max_index, 'Source']))
        frequency_locations.append(int(data_copy.at[max_index, 'Frequency']))
        data_copy = data_copy.drop(max_index)
        data_copy['Frequency'] = data_copy.iloc[:, 1:-2].sum(axis=1)
    #print(frequency_norm)
    if normal==True:
        frequency_values = np.array(frequency_locations)
        # frequency_norm = (frequency_values - frequency_values.min()) / (frequency_values.max() - frequency_values.min())
        frequency_norm = frequency_values / frequency_values.max()
        frequency=list(map(float, frequency_norm))
    else:
        frequency=list(map(int, frequency_locations))

    return list(map(int, sensor_locations)),frequency
# Fungsi untuk Heuristik Preprocessing
def heuristic_preprocessing(graph, combined_centrality,label, threshold=0.7):
    # Pilih node dengan nilai combined centrality di atas threshold tertentu
    return [node for node, centrality in combined_centrality.items() if centrality >= threshold]
# Fungsi Optimasi Lokal
def local_optimization(graph, initial_placement, combined_centrality,label):
    current_placement = initial_placement.copy()
    improved = True
    current_evaluate=evaluate_placement(graph, current_placement, combined_centrality)
    while improved:
        improved = False
        for node in set(graph.nodes()) - set(current_placement):
            if node in combined_centrality.keys():
                for sensor_node in current_placement:
                    new_placement = current_placement.copy()
                    new_placement.remove(sensor_node)
                    new_placement.append(node)
                    # Evaluasi apakah penggantian meningkatkan solusi
                    if evaluate_placement(graph, new_placement, combined_centrality) > current_evaluate:
                        current_placement = new_placement
                        improved = True
    return current_placement
# Fungsi Evaluasi Penempatan Sensor (contoh sederhana, bisa disesuaikan dengan kriteria yang lebih kompleks)
def evaluate_placement(graph, placement, combined_centrality):
    return sum(combined_centrality[int(node)] for node in placement)

def timeEstimated(steps,readLink,sensor_location,label):
    # steps=steps[1:]
    readLink=readLink.loc[readLink['Source'].isin(sensor_location)]
    tr={}
    for index,item in readLink.iterrows():
        stepPos=item.loc[steps]
        if (stepPos == 1).any():
            first_position_index = stepPos[stepPos == 1].index[0]
            first_position_index_numeric = stepPos.index.get_loc(first_position_index)
        else:
            first_position_index_numeric=None
        tr[item['Source']]=first_position_index_numeric
    reposition=[tr[k] for k in sensor_location]
    # reposition=[]
    # first=True
    # for k in sensor_location:
    #     val=tr[k]
    #     if val==0 and first!=True:
    #         val=None
    #     reposition.append(val)
    #     first=False
    return reposition

directory='source_inp/output_simulation'
pathcsv=f'{directory}/time_contamination'
networkset=['fos']#,'bwsn'
listNetwork=['FOS - unvertices.inp']#,'BWSN-clean.inp'
num_sensors = [2,3,4,5] #,

for i,ns in enumerate(networkset):
    N= epanet(f'source_inp/data_network/{listNetwork[i]}')
    N.plot_close()
    csvfile = open(f"{directory}/{ns}_9.0_greedy.csv", "w",newline='')
    csvfileevbw = open(f"{directory}/{ns}_9.0_greedy_ev_bw.csv", "w",newline='')
    csvfileevbwh = open(f"{directory}/{ns}_9.0_greedy_ev_bw_heurastic.csv", "w",newline='')
    writertank = csv.writer(csvfile, dialect='excel',delimiter=";")
    writertankevbw = csv.writer(csvfileevbw, dialect='excel',delimiter=";")
    writertankevbwh = csv.writer(csvfileevbwh, dialect='excel',delimiter=";")

    readLink = pd.read_csv(f'{pathcsv}/{ns}_link.csv',on_bad_lines='skip',delimiter=';')
    
    nodeSource=[i for i in N.getLinkIndex()]
    steps = [col for col in readLink.columns if col.startswith('links_step')]
    step_data = {col: np.array([x if pd.isna(x)!=True else 0 for x in readLink[col]]) for col in steps}
    head=['Node Contaminant','Jumlah Sensor','Algorithm','Sensor Placement','Score','Time estimated']
    writertank.writerow(head)
    writertankevbw.writerow(head)
    writertankevbwh.writerow(head)
    for id,label in enumerate(range(1,len(readLink['NodeID'])+1)):
        resultLink=readDataFrame(readLink,label,nodeSource)
        resultLinkO=readDataFrame(readLink,label,nodeSource,valDefault=None)
        netwz=nx.Graph()
        combine_centrality=combine_centrality_compute(netwz,step_data,label,normal=True)
        candidate_nodes = heuristic_preprocessing(netwz, combine_centrality,label, threshold=0.7)
        for sensor in num_sensors :
            sensor_placement,score = greedy_sensor_placement(netwz, combine_centrality, sensor)
            sp_ori,score_ori = greedy_sensor_placement_original(resultLink, sensor,label,normal=True)
            
            initial_placement,initial_score = greedy_sensor_placement(netwz.subgraph(candidate_nodes), combine_centrality, sensor)
            final_placement = local_optimization(netwz, initial_placement, combine_centrality,label)
            time_ori=timeEstimated(steps,resultLinkO,sp_ori,label)
            time_gevbw=timeEstimated(steps,resultLinkO,sensor_placement,label)
            time_hegevbw=timeEstimated(steps,resultLinkO,final_placement,label)
            writertank.writerow([label]+[sensor]+['Greedy']+[sp_ori]+[score_ori]+[time_ori])
            writertankevbw.writerow([label]+[sensor]+['Greedy+ev+bw']+[sensor_placement]+[score]+[time_gevbw])
            writertankevbwh.writerow([label]+[sensor]+['Heurastic:Greedy+ev+bw']+[final_placement]+[initial_score]+[time_hegevbw])
    csvfile.close()
    csvfileevbw.close()
    csvfileevbwh.close()