from epyt import epanet
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import networkx as nx
import time
import pandas as pd
import ast

G = epanet(f'source_inp/data_network/BWSN-clean.inp')
# G = epanet(f'source_inp/data_network/FOS - unvertices.inp')
name_network="bwsn"
# name_network="fos"
G.plot_close()
#=========================================================
def get_first(iterable, value=None, key=None, default=None):
    match value is None, callable(key):
        case (True, True):
            gen = (elem for elem in iterable if key(elem))
        case (False, True):
            gen = (elem for elem in iterable if key(elem) == value)
        case (True, False):
            gen = (elem for elem in iterable if elem)
        case (False, False):
            gen = (elem for elem in iterable if elem == value)

    return next(gen, default)
#=========================================================
def QualityContaminant(ql,qt,stimer):
    result=[]
    for ik,vk in enumerate(ql):
        res=[]
        for qlkey,qlval in enumerate(vk):
            if(round(qlval,2)>qt):
                res.append(qlkey+1)
        if(len(res)):
            result.append(res)
    lastItem=result[len(result)-1]
    first=True
    for key,rval in enumerate(result):
        if first==True and rval==lastItem:
            result=[len(rval),key]
            first=False
    return result
#=========================================================
G.setQualityType('chem', 'Chlorine', 'mg/L')
simulationDuration = 18000    #Duration-> 172800 seconds = 2days | default => 86400 = 1 days
G.setTimeSimulationDuration(simulationDuration)
patternStart = 0 # Pattern Start
G.setTimePatternStart(patternStart)
patternStep = 1 # Pattern Timestep
G.setTimePatternStep(patternStep)
G.setTimeReportingStart(patternStart)
G.setTimeReportingStep(patternStep)
# G.setFlowUnitsGPM()
#=========================================================
directory="source_inp/output_simulation/report_jurnal" #Colab
if not os.path.exists(directory):
    os.makedirs(directory)
start_time = time.time()
#=========================================================
# def eigenvector_centrality_normalize(st,std):
def eigenvector_centrality(data,steps,step_data):
    n_nodes = len(data['NodeID'])
    
    # Build adjacency matrix
    adjacency_matrix = np.zeros((n_nodes, n_nodes))
    for step in steps:
        for i, nodes in enumerate(step_data[step]):
            for node in nodes:
                if node > 0 and node < n_nodes:
                    adjacency_matrix[i % (n_nodes-1) + 1][node] = 1
                    adjacency_matrix[node][i % (n_nodes-1) + 1] = 1

    # Compute eigenvector centrality
    eigenvalues, eigenvectors = np.linalg.eig(adjacency_matrix)
    max_eigenvalue_index = np.argmax(eigenvalues)
    eigenvector_centralities = np.abs(eigenvectors[:, max_eigenvalue_index])
    
    # Normalize eigenvector centralities
    max_value = np.max(eigenvector_centralities)
    if max_value > 0:
        eigenvector_centralities /= max_value
    # Create a dictionary to hold centrality values
    centrality_dict = {i: eigenvector_centralities[i] for i in range(0, n_nodes)}
    return centrality_dict
#=========================================================
kmax=0 # batas aman kontaminant
plotReport=1 #ouput plot setiap :h jam
plotStepTimer=plotReport*60*patternStep
plotTimer=1*60*patternStep
#=========================================================
#hitung eg_centrality
#=============================================
file_path = f'source_inp/output_simulation/time_contamination/{name_network}_node.csv'
data = pd.read_csv(file_path,on_bad_lines='skip',delimiter=';')

steps = [col for col in data.columns if col.startswith('node_step')]
step_data = {col: np.array([set(eval(x)) for x in data[col]]) for col in steps}
egcentrality = eigenvector_centrality(data,steps,step_data)
#================================================
NL,LL={},{}
for ik,vk in enumerate(G.getNodeIndex()):
    nodeIndex=vk #15
    sourceType = 'SETPOINT' #MASS,CONCEN,SETPOINT,FLOWPACED
    G.setNodeSourceType(nodeIndex, sourceType)
    s=[]
    sourceStrength=1.5 #besaran kontaminant yang akan di injeksi (yang merubah setingan default dari Network yang digunakan)
    for ikz,vkz in enumerate(G.getNodeIndex()):
        if vkz == nodeIndex:
            s.append(sourceStrength)
        else:
            s.append(0.0)
    G.setNodeInitialQuality(G.getNodeIndex(),s)
    G.setNodeSourceQuality(G.getNodeIndex(),s)
    tstep, N,L = 1, [],[]
    G.openHydraulicAnalysis()
    G.openQualityAnalysis()
    G.initializeHydraulicAnalysis(0)
    G.initializeQualityAnalysis(G.ToolkitConstants.EN_NOSAVE)
    while tstep>0:
        t = G.runHydraulicAnalysis()
        qt = G.runQualityAnalysis()
        if int(t)%plotStepTimer== 0 :
            QL=G.getLinkActualQuality()
            QN=G.getNodeActualQuality()
            N.append(QN)
            L.append(QL)
        #memanggil Analisa berikutnya (agar looping bisa berjalan)
        tstep=G.nextHydraulicAnalysisStep()
        qtstep = G.nextQualityAnalysisStep()
    #menutup proses analisa
    G.closeHydraulicAnalysis()
    G.closeQualityAnalysis()
    NL[vk]=N
    LL[vk]=L
#=========================================================
with open(f'{directory}/{name_network}_node.csv', 'w',newline='') as csvfile:
    writertank = csv.writer(csvfile, dialect='excel',delimiter=";")
    header=['NodeID']+['Frequency/Coverage','time','eg_centrality']
    writertank.writerow(header)
    for it,itv in enumerate(G.getNodeIndex()):
        qcNode=QualityContaminant(NL[itv],kmax,plotStepTimer)
        rowdata=[itv] + qcNode + [round(egcentrality[it], 7)]
        writertank.writerow(rowdata)
