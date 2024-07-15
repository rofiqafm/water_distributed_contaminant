import pandas as pd
import ast
import csv
import numpy as np
import networkx as nx

def compute_eigenvector_centrality(data,max_iter=500):
    """
    Menghitung eigenvector centrality untuk setiap node dalam jaringan.

    Args:
    data: DataFrame yang berisi sumber kontaminan dan pergerakan kontaminan setiap interval waktu.
    max_iter: Jumlah maksimum iterasi untuk konvergensi.

    Returns:
    centrality: Dictionary dengan node sebagai key dan eigenvector centrality sebagai value.
    """
    G = nx.Graph()
    for index, row in data.iterrows():
        for col in data.columns[1:]:
            if row[col] > 0:
                G.add_edge(str(row['Source']), col)
    
    if len(G) == 0:
        print("Graph is empty, returning empty centrality.")
        return {}
    
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        print(f"Power iteration failed to converge within {max_iter} iterations.")
        centrality = {}
    return centrality

def greedy_sensor_placement(data, num_sensors,centrality):
    """
    Algoritma greedy untuk penempatan sensor deteksi kontaminan.

    Args:
    data: DataFrame yang berisi sumber kontaminan dan pergerakan kontaminan setiap interval waktu.
          Kolom pertama adalah sumber kontaminan, kolom berikutnya menunjukkan pergerakan kontaminan setiap interval waktu.
    num_sensors: Jumlah sensor yang akan ditempatkan.

    Returns:
    sensor_locations: Daftar lokasi sensor terbaik berdasarkan algoritma greedy.
    """

    # Langkah 1: Menghitung frekuensi setiap node terkena kontaminan
    data_copy = data.copy()
    data_copy['Frequency'] = data_copy.iloc[:, 1:].sum(axis=1)
    
    # Ubah tipe data 'Source' menjadi string untuk kesesuaian dengan centrality
    data_copy['Source'] = data_copy['Source'].astype(str)

     # Kombinasi frekuensi dengan centrality
    data_copy['Score'] = data_copy.apply(lambda row: row['Frequency'] * centrality.get(row['Source'], 0), axis=1)
    
    # Inisialisasi hasil dan urutkan berdasarkan skor gabungan
    result = {
        'sensor_locations': [],
        'dataset': data_copy[['Source','Frequency', 'Score']].sort_values(by=['Score','Frequency', 'Source'], ascending=[False, False, True])
    }
    
    # Langkah 2: Inisialisasi daftar lokasi sensor
    sensor_locations = []
    # Langkah 3: Algoritma greedy untuk memilih lokasi sensor
    for _ in range(num_sensors):
        # Pilih node dengan skor tertinggi
        max_index = data_copy['Score'].idxmax()
        sensor_locations.append(int(data_copy.at[max_index, 'Source']))

        # Hapus node yang sudah dipilih sebagai sensor
        data_copy = data_copy.drop(max_index)

        # Hitung ulang skor setelah menghapus node yang sudah dipilih
        data_copy['Frequency'] = data_copy.iloc[:, 1:-2].sum(axis=1)
        data_copy['Score'] = data_copy.apply(lambda row: row['Frequency'] * centrality.get(row['Source'], 0), axis=1)

    result['sensor_locations']=list(map(int, sensor_locations))
    return result

# Calculate coverage
def calculate_coverage(sensor_positions, step_data,steps):
    n_nodes = len(step_data[steps[0]])
    total_detected = np.zeros(n_nodes, dtype=int)
    for step in steps:
        detected = np.zeros(n_nodes, dtype=int)
        for sensor in sensor_positions:
            detected |= [str(sensor) in step_data[step][i] for i in range(n_nodes)]
        total_detected |= detected
    # coverage = np.sum(total_detected) / n_nodes * 100  
    coverage = np.sum(total_detected)
    return coverage

#load csv hasil generate time_contamination
# pathcsv='source_inp/time_contamination'
directory='source_inp/output_simulation'
pathcsv=f'{directory}/time_contamination'
networkset=['fos','bwsn']

for i,ns in enumerate(networkset):
    csvfile = open(f"{directory}/{ns}_8.1_report.csv", "w",newline='')
    writertank = csv.writer(csvfile, dialect='excel',delimiter=";")
    header=['Contaminant(Node)','Sensor2(Link)','Coverage2','Sensor3(Link)','Coverage3','Sensor4(Link)','Coverage4','Sensor5(Link)','Coverage5']
    readLink = pd.read_csv(f'{pathcsv}/{ns}_link.csv',on_bad_lines='skip',delimiter=';')
    nodeSource=[i for i in range(1,len(readLink['NodeID'])+1)]
    writertank.writerow(header)

    steps = [col for col in readLink.columns if col.startswith('links_step')]
    step_data = {col: np.array([x if pd.isna(x)!=True else 0 for x in readLink[col]]) for col in steps}

    for id,label in enumerate(range(1,len(readLink['NodeID'])+1)):
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
                        val=0
                        if pd.isna(row[keys])!=True:
                            res=ast.literal_eval(row[keys])
                            if node in res:
                                val=1
                        dataResultLink[keys].append(val)
        resultLink=pd.DataFrame(dataResultLink)
        # Hitung eigenvector centrality dengan iterasi maksimum lebih tinggi
        centrality = compute_eigenvector_centrality(resultLink, max_iter=1000)
        print(centrality.keys())
        exit()
        num_sensors = [2,3,4,5]
        sensor_list=[]
        for sensor in num_sensors :
            sensor_locationsLink = greedy_sensor_placement(resultLink, sensor,centrality)
            coverage = calculate_coverage(sensor_locationsLink['sensor_locations'],step_data,steps)
            sensor_list.append(sensor_locationsLink['sensor_locations'])
            sensor_list.append(coverage)
        row=[label]+sensor_list
        writertank.writerow(row)
    csvfile.close()