import pandas as pd
import ast

def greedy_sensor_placement(data, num_sensors):
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
    # Hilangkan kolom pertama karena itu adalah sumber kontaminan
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
    result['sensor_locations']=sensor_locations
    return result

#load csv hasil generate time_contamination
# pathcsv='source_inp/time_contamination'
directory='source_inp/output_simulation'
pathcsv=f'{directory}/time_contamination'
networkset=['fos','jilin']
networknodeset=[37,28]
for i,ns in enumerate(networkset):
    f = open(f"{directory}/{ns}_sensor_placement.txt", "w",newline='')
    read = pd.read_csv(f'{pathcsv}/{ns}_node.csv',on_bad_lines='skip',delimiter=';')
    readLink = pd.read_csv(f'{pathcsv}/{ns}_link.csv',on_bad_lines='skip',delimiter=';')

    #menentukan Kontaminasi Node Source
    nodeContaminantSouce=networknodeset[i]
    # nodeSource=[35,36,37]
    # nodeSource=[35,36,37, 54, 58]
    nodeSource=[i for i in range(1,networknodeset[i]+1)]

    dataResult={'Source':[]}
    dataResultLink={'Source':[]}
    dataCopy=read.loc[[nodeContaminantSouce-1]].copy()
    dataCopyLink=readLink.loc[[nodeContaminantSouce-1]].copy()

    #perulangan berdasarakn data csv Node
    for index,row in dataCopy.iterrows():
        #perulangan berdasarakan jumlah node source
        for kr,ir in enumerate(nodeSource):
            dataResult['Source'].append(ir)
            node=ir
            #perulangan setiap colomn pada setiap data untu melakukan pengencekan kontaminasi pada setiap node
            for keys in dataCopy.keys():
                if keys!='NodeID' :
                    if keys not in dataResult:
                        dataResult[keys]=[]
                    val=0
                    #cek data is not NaN
                    if pd.isna(row[keys])!=True:
                        #merubah data string AST ke dalam bentuk tipe data list
                        res=ast.literal_eval(row[keys])
                        #cek node terjadi kontaminasi atau tidak
                        if node in res:
                            val=1
                    dataResult[keys].append(val)
    #merubah data list ke bentuk tipe data : dataFrame
    result=pd.DataFrame(dataResult)

    #perulangan berdasarakn data csv Link
    for index,row in dataCopyLink.iterrows():
        #perulangan berdasarakan jumlah node source
        for kr,ir in enumerate(nodeSource):
            dataResultLink['Source'].append(ir)
            node=ir
            #perulangan setiap colomn pada setiap data untu melakukan pengencekan kontaminasi pada setiap node
            for keys in dataCopyLink.keys():
                if keys!='NodeID' :
                    if keys not in dataResultLink:
                        dataResultLink[keys]=[]
                    val=0
                    #cek data is not NaN
                    if pd.isna(row[keys])!=True:
                        #merubah data string AST ke dalam bentuk tipe data list
                        res=ast.literal_eval(row[keys])
                        #cek node terjadi kontaminasi atau tidak
                        if node in res:
                            val=1
                    dataResultLink[keys].append(val)
    #merubah data list ke bentuk tipe data : dataFrame
    resultLink=pd.DataFrame(dataResultLink)

    # Jumlah sensor yang diinginkan
    num_sensors = [2,3,4,5]
    f.write("=====================================")
    f.write(f"\n|    Network {ns.upper()}:Algoritma Greedy    |")
    f.write("\n=====================================")
    for sensor in num_sensors :
        # Menentukan lokasi sensor terbaik
        sensor_locations = greedy_sensor_placement(result, sensor)
        f.write(f"\nJumlah Sensor:{sensor}| Lokasi Sensor Terbaik pada Node:{sensor_locations['sensor_locations']}")
        f.write("\n-----------------------------------------------------")
        f.write(f"\n{sensor_locations['dataset'].to_string(header=True, index=False)}")
        sensor_locationsLink = greedy_sensor_placement(resultLink, sensor)
        f.write("\n-----------------------------------------------------")
        f.write(f"\nJumlah Sensor:{sensor}| Lokasi Sensor Terbaik pada Link:{sensor_locationsLink['sensor_locations']}")
        f.write("\n-----------------------------------------------------")
        f.write(f"\n{sensor_locationsLink['dataset'].to_string(header=True, index=False)}")
        if sensor!=num_sensors[len(num_sensors)-1]:
            f.write("\n-----------------------------------------------------")
    f.write("\n")
    f.close()