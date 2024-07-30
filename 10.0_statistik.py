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

def readDataFrameAll(readLink,nodeSource,valDefault=0):
# =====================================
    #DataFrame untuk Frequency greedy original
    dataResultLink={'Source':[]}
    dataCopyLink=readLink.copy()
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

def greedy_sensor_placement(data, num_sensors):
    data_copy = data.copy()
    data_copy['Frequency'] = data_copy.iloc[:, 1:].sum(axis=1)
    dataset=data_copy[['Source','Frequency']].sort_values(by=['Frequency','Source'], ascending=[False,True])
    group=dataset.groupby(['Source'])['Frequency'].max().reset_index().sort_values(by=['Frequency','Source'], ascending=[False,True])
    group_copy = group.copy()
    sensor_locations = []
    for _ in range(num_sensors):
        max_index = group_copy['Frequency'].idxmax()
        sensor_locations.append(group_copy.at[max_index, 'Source'])
        group_copy = group_copy.drop(max_index)
    return sensor_locations,group

def greedy_sensor_placementsum(data, num_sensors):
    data_copy = data.copy()
    data_copy['Frequency'] = data_copy.iloc[:, 1:].sum(axis=1)
    dataset=data_copy[['Source','Frequency']].sort_values(by=['Frequency','Source'], ascending=[False,True])
    group=dataset.groupby(['Source'])['Frequency'].sum().reset_index().sort_values(by=['Frequency','Source'], ascending=[False,True])
    group_copy = group.copy()
    sensor_locations = []
    for _ in range(num_sensors):
        max_index = group_copy['Frequency'].idxmax()
        sensor_locations.append(group_copy.at[max_index, 'Source'])
        group_copy = group_copy.drop(max_index)
    return sensor_locations,group

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
    readLink = pd.read_csv(f'{pathcsv}/{ns}_link.csv',on_bad_lines='skip',delimiter=';')
    readLinkQuality = pd.read_csv(f'{pathcsv}/{ns}_link_quality.csv',on_bad_lines='skip',delimiter=';')
    
    nodeSource=[i for i in N.getLinkIndex()]
    steps = [col for col in readLink.columns if col.startswith('links_step')]
    step_data = {col: np.array([x if pd.isna(x)!=True else 0 for x in readLink[col]]) for col in steps}
    all_values = []
    for key, values in step_data.items():
        for val in values:
            all_values.extend(val[1:-1].split(","))
    all_values = [int(x.strip()) for x in all_values if x!='']
    unique_values = list(set(all_values))
    
    dataresult={}
    dataresult['Source']={}
    for id,label in enumerate(range(1,len(readLink['NodeID'])+1)):
        resultLinkO=readDataFrame(readLink,label,nodeSource,valDefault=None)
        print(readLinkQuality.keys())
        exit()
        dataresult[label]={}
        # dataresult['Source']='Source'
        for link in unique_values:
            dataresult[label]['Links']=label
            dataresult['Source'][link]=''
            dataresult[label][link]=timeEstimated(steps,resultLinkO,[link],label)[0]

    column_data=pd.DataFrame(dataresult)
    transpose_data=column_data.transpose()
    listColumn=list(transpose_data.columns.values)
    transpose_data=transpose_data[[listColumn[-1]]+listColumn[:-1]]
    # print(transpose_data)
    # exit()
    all_result=readDataFrameAll(readLink,nodeSource)
    gsensor,set=greedy_sensor_placement(all_result, 1)
    gsensorsum,setsum=greedy_sensor_placementsum(all_result, 1)
    file_name_export = f"{directory}/{ns}_10.0_statistik.xlsx"
    with pd.ExcelWriter(file_name_export) as writer:
        head=['Node Contaminant','Jumlah Sensor','Algorithm','Sensor Placement','Score','Time estimated']
        transpose_data.style.to_excel(writer, sheet_name='Data Matrix', index=False)
        set.style.to_excel(writer, sheet_name='Greedy Result(Max)', index=False)
        setsum.style.to_excel(writer, sheet_name='Greedy Result(Sum)', index=False)