from epyt import epanet
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from PIL import Image

# G = epanet('networks/Jilin including water quality.inp') #local
# G = epanet('source_inp/network/Jilin including water quality.inp') #Colab
# G = epanet('source_inp/FOS.inp') #Colab
# G = epanet('source_inp/data_network/FOS - unvertices.inp')
G = epanet(f'source_inp/data_network/BWSN-clean.inp')
# G = epanet('source_inp/data_network/Jilin including water quality.inp')
name_network="bwsn"
G.plot_close()
#=========================================================
def QualityContaminant(ql,qt,s):
    result=[]
    for ik,vk in enumerate(ql):
        res=[]
        for qlkey,qlval in enumerate(vk):
            if(round(qlval,2)>qt):
                res.append(qlkey+1)
        if(len(res)):
            result.append(res)
    return result

def calcTContaminant(qcl,nTarget):
    result=None
    sF=True
    for keyCtc,valCtc in enumerate(qcl):
        if valCtc.count(nTarget) and sF==True:
            result=keyCtc
            sF=False
    return result
#=========================================================
G.setQualityType('chem', 'Chlorine', 'mg/L')
simulationDuration = 18000    #Duration-> 172800 seconds = 2days | default => 86400 = 1 days
G.setTimeSimulationDuration(simulationDuration)
patternStart = 0 # Pattern Start
G.setTimePatternStart(patternStart)
patternStep = 5 # Pattern Timestep
G.setTimePatternStep(patternStep)
G.setTimeReportingStart(patternStart)
G.setTimeReportingStep(patternStep)
# G.setFlowUnitsGPM()
#=========================================================
# directory="time_contamination" #local
directory="source_inp/output_simulation/time_contamination" #Colab
if not os.path.exists(directory):
    os.makedirs(directory)
#=========================================================
kmax=0 # batas aman kontaminant
plotReport=1 #ouput plot setiap :h jam
plotStepTimer=plotReport*60*patternStep
plotTimer=1*60*patternStep
#=========================================================
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
nodeContaminantTimer=None
nodeSource=129
nodeTarget=7
linkContaminantTimer=None
linkTarget=None #None or linkID
fextra='' #"_15 kosong"
#=========================================================
with open(f'{directory}/{name_network}_node{fextra}.csv', 'w',newline='') as csvfile:
    writertank = csv.writer(csvfile, dialect='excel',delimiter=";")
    header=['NodeID']+[f'node_step{i*patternStep}' for i in range(len(NL[1]))]
    writertank.writerow(header)
    fcNode=True
    for it,itv in enumerate(G.getNodeIndex()):
        qcNode=QualityContaminant(NL[itv],kmax,itv)
        if itv==nodeSource and fcNode==True and nodeTarget!=None:
            nodeContaminantTimer=calcTContaminant(qcNode,nodeTarget)
            fcNode=False
        rowdata=[itv] + qcNode
        writertank.writerow(rowdata)
with open(f'{directory}/{name_network}_link{fextra}.csv', 'w',newline='') as csvfile:
    writertank = csv.writer(csvfile, dialect='excel',delimiter=";")
    header=['NodeID']+[f'links_step{i}' for i in range(len(LL[1]))]
    writertank.writerow(header)
    fcLink=True
    for il,ilv in enumerate(G.getNodeIndex()):
        qcLink=QualityContaminant(LL[ilv],kmax,ilv)
        if ilv==nodeSource and fcLink==True and linkTarget!=None:
            linkContaminantTimer=calcTContaminant(qcLink,linkTarget)
            fcLink=False
        rowdata=[ilv] +qcLink
        writertank.writerow(rowdata)

statisticsType = 'NONE'
G.setTimeStatisticsType(statisticsType)
G.setReport('NODES ALL')
G.setReport('LINKS ALL')
#=========================================================
if nodeContaminantTimer !=None:
  G.plot(
      node_values=NL[nodeSource][nodeContaminantTimer],
      link_values=LL[nodeSource][nodeContaminantTimer],
      title=f'Source Contaminant Node:{nodeSource} Waktu Ke : {nodeContaminantTimer*patternStep}m sampai Node:{nodeTarget}',
      fig_size=[3,3],
      fontsize=5,
      figure=False,
      node_text=True,
      link_text=True
      # linksID=True,
      # nodesID=True,
  )
  G.plot_save(f'{directory}/nodeTargetContaminantA.png')
  G.plot_close()
  G.plot(
      nodesID=True,
      linksID=True,
      fig_size=[3,3],
      fontsize=3,
      figure=False,
      title=f'Source Contaminant Node:{nodeSource} Waktu Ke : {nodeContaminantTimer*patternStep}m sampai Node:{nodeTarget}'
  )
  G.plot_save(f'{directory}/nodeTargetContaminantB.png')
  G.plot_close()
  #=========================================================
  listImage=[f'{directory}/nodeTargetContaminantA.png',
              f'{directory}/nodeTargetContaminantB.png']
  images = [Image.open(x) for x in listImage]
  widths, heights = zip(*(i.size for i in images))
  total_width = sum(widths)
  max_height = max(heights)
  new_im = Image.new('RGB', (total_width, max_height))
  x_offset = 0
  for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

  new_im.save(f'{directory}/{name_network}_nodeTargetContaminant{fextra}.png')
  os.remove(f'{directory}/nodeTargetContaminantA.png')
  os.remove(f'{directory}/nodeTargetContaminantB.png')
#=========================================================
  print("==================================================================")
  print(f'Source Contaminant Node:{nodeSource} membutuhkan waktu {nodeContaminantTimer*patternStep}m untuk sampai pada Node:{nodeTarget}')
  print("==================================================================")
#=========================================================
if linkTarget!=None and linkContaminantTimer!=None:
    G.plot(
        node_values=NL[nodeSource][linkContaminantTimer],
        link_values=LL[nodeSource][linkContaminantTimer],
        title=f'Source Contaminant Node:{nodeSource} Waktu Ke : {linkContaminantTimer*patternStep}m sampai Link:{linkTarget}',
        fig_size=[3,3],
        fontsize=5,
        figure=False,
        node_text=True,
        link_text=True
        # linksID=True,
        # nodesID=True,
    )
    G.plot_save(f'{directory}/linkTargetContaminantA.png')
    G.plot_close()
    G.plot(
        nodesID=True,
        linksID=True,
        fig_size=[3,3],
        fontsize=3,
        figure=False,
        title=f'Source Contaminant Node:{nodeSource} Waktu Ke : {linkContaminantTimer*patternStep}m sampai Link:{linkTarget}'
    )
    G.plot_save(f'{directory}/linkTargetContaminantB.png')
    G.plot_close()
    #=========================================================
    listImage=[f'{directory}/linkTargetContaminantA.png',
                f'{directory}/linkTargetContaminantB.png']
    images = [Image.open(x) for x in listImage]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(f'{directory}/{name_network}_linkTargetContaminant{fextra}.png')
    os.remove(f'{directory}/linkTargetContaminantA.png')
    os.remove(f'{directory}/linkTargetContaminantB.png')
    #=========================================================
    print("==================================================================")
    print(f'Source Contaminant Node:{nodeSource} membutuhkan waktu {linkContaminantTimer*patternStep}m untuk sampai pada Links:{linkTarget}')
    print("==================================================================")

P=G.runsCompleteSimulation()
