import numpy as np
import pandas as pd
import random
import time
import ast

# Load the CSV file with a semicolon delimiter
directory='source_inp/output_simulation'
file_path = f'{directory}/time_contamination/fos_link.csv'
data = pd.read_csv(file_path,on_bad_lines='skip',delimiter=';')
nodeSource=[i for i in range(1,len(data['NodeID'])+1)]
# Process the data
steps = [col for col in data.columns if col.startswith('links_step')]
step_data = {col: np.array([x if pd.isna(x)!=True else 0 for x in data[col]]) for col in steps}
dataResultLink={'Source':[]}
dataCopyLink=data.loc[:].copy()
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

def greedy_sensor_placement(data, num_sensors):
    data_copy = data.copy()
    data_copy['Frequency'] = data_copy.iloc[:, 1:].sum(axis=1)
    #inisiasi result dan mengambil hasil bobot dan mengurutkan berdasarkan besaran nya dan urutan Node/Link
    result={
        'sensor_locations':[],
        'dataset':data_copy[['Source','Frequency']].sort_values(by=['Frequency','Source'], ascending=[False,True])
    }
    # Langkah 2: Inisialisasi daftar lokasi sensor
    sensor_locations = []
    # Langkah 3: Algoritma greedy untuk memilih lokasi sensor
    for _ in range(num_sensors):
        # Pilih node dengan frekuensi terkena kontaminan tertinggi
        max_index = data_copy['Frequency'].idxmax()
        sensor_locations.append(data_copy.at[max_index, 'Source'])

        # Hapus node yang sudah dipilih sebagai sensor
        data_copy = data_copy.drop(max_index)

        # Hitung ulang frekuensi terkena kontaminan setelah menghapus node yang sudah dipilih
        data_copy['Frequency'] = data_copy.iloc[:, 1:-1].sum(axis=1)
    result['sensor_locations']=list(map(int, sensor_locations))
    return result

num_sensors = [2,3]
results=[]
for sensor in num_sensors:
    # Run and time each algorithm
    start_time = time.time()
    greedy_positions = greedy_sensor_placement(data, sensor)
    greedy_time = time.time() - start_time
    greedy_coverage=calculate_coverage(greedy_positions, step_data)
    txt=f"Greedy Algorithm: Positions={greedy_positions}, Coverage={greedy_coverage:.2f}%, Time={greedy_time:.2f} seconds"
    results.append(txt)
print(results)
    