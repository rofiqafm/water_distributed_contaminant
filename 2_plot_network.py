from epyt import epanet
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import networkx as nx

#load File INP
# G = epanet('source_inp/FOS.inp')
# G = epanet('source_inp/data_network/FOS - unvertices.inp')
G = epanet('source_inp/data_network/Jilin including water quality.inp')
# G = epanet('source_inp/network/New York Tunnels including water quality.inp')
# G.loadMSXFile('source_inp/net2-cl2.msx')

name_network="jilin"
#Menutup semua Plot Network yang terbuka
G.plot_close()

#fungis konversi timestamp ke format 00:00:00
def convert_seconds(seconds):
    units = {"hours": 3600, "minutes": 60, "seconds": 1}
    values = []
    for unit, value in units.items():
        count = seconds // value
        seconds -= count * value
        values.append(count)
    return f"{values[0]:02d}:{values[1]:02d}:{values[2]:02d}"

G.setQualityType('chem', 'Chlorine', 'mg/L')
#setting lama durasi simulasi
simulationDuration = 18000    #Duration-> 172800 seconds = 2days | default => 086400 = 1 days
G.setTimeSimulationDuration(simulationDuration)
patternStart = 0 # Pattern Start
G.setTimePatternStart(patternStart)
patternStep = 1 # Pattern Timestep
G.setTimePatternStep(patternStep)
G.setTimeReportingStart(patternStart)
G.setTimeReportingStep(patternStep)

listNode=[27,28]
for kln,vln in enumerate(listNode):
  nodeIndex=vln
  sourceType = 'SETPOINT' #MASS,CONCEN,SETPOINT,FLOWPACED
  G.setNodeSourceType(nodeIndex, sourceType)

  #fungsi untuk mengadjust default injeksi pada network
  s=[]
  nodeIndexA=[nodeIndex] #index node yang akan di injeksi, ketika lebih dari satu node maka tambahkan index node dalam bentuk array
  sourceStrength=1.5 #besaran kontaminant yang akan di injeksi (yang merubah setingan default dari Network yang digunakan)
  for ik,vk in enumerate(G.getNodeIndex()):
      if vk in nodeIndexA:
          s.append(sourceStrength)
      else:
          s.append(0.0)
  G.setNodeInitialQuality(G.getNodeIndex(),s)
  G.setNodeSourceQuality(G.getNodeIndex(),s)
  # print(G.getNodeInitialQuality(nodeIndexA))

  #membuat direktori source_inp/plot untuk menaruh menyimpan file plot network yang akan di generate
  directory=f"source_inp/output_simulation/plot_network/{name_network}/node_{nodeIndex}"
  # shutil.rmtree(directory)
  if not os.path.exists(directory):
      os.makedirs(directory)

  #Menjalankan simulasi
  G.openHydraulicAnalysis()
  G.openQualityAnalysis()
  G.initializeHydraulicAnalysis(0)
  G.initializeQualityAnalysis(G.ToolkitConstants.EN_NOSAVE)

  kmax=0 # batas aman kontaminant
  kmaxFlow=0.0
  plotReport=1 #ouput plot setiap :h jam

  plotStepTimer=plotReport*60*patternStep
  plotTimer=1*60*patternStep
  tstep, P,T = 1, [],[]
  #looping pada setiap waktu untuk melakukan generate Plot network dalam bentuk image (png)
  while tstep>0:
      t = G.runHydraulicAnalysis()
      qt = G.runQualityAnalysis()
      P.append(G.getNodeIndex())
      T.append(t)
      if int(t)%plotStepTimer== 0 :
          # QN=G.getNodeActualQuality()
          QN=G.getNodeActualQuality() #mengambil nilai quality/kontaminasi pada semua node pada waktu aktual (terjadinya looping)
          if t!=0:
            QL=G.getLinkActualQuality() #mengambil nilai quality/kontaminasi pada semua link pada waktu aktual (terjadinya looping)
          else:
            QL=[0 for xq in G.getLinkActualQuality()]

          # Vn=G.getNodePressure()
          Vl=G.getLinkFlows() #mengambil besaran FLow pada semua link pada waktu aktual (terjadinya looping)
          HN,HL,NV,HV=[],[],[],[] #inisialisasi variable
          namelabel=convert_seconds(t) #merubah timestamp untuk labeling judul pada gambar
          filename=int(t/plotTimer) #nama file yang akan di export menggunakan waktu aktual (terjadinya looping)
          #Looping untuk mendeteksi besaran quality/kontaminasi yang melebihi batas aman pada node
          for key, nQ in enumerate(QN):
              if(nQ>kmax):
                  HN.append(key+1)
                  # NV.append(nQ)
          #Looping untuk mendeteksi besaran quality/kontaminasi yang melebihi batas aman pada link
          for keyL, nL in enumerate(QL):
              if(nL>kmax and t!=0):
                  HL.append(keyL+1)
          #Looping untuk mendeteksi besaran Flow yang terjadi Negative FLow < 0
          for keyF, nF in enumerate(Vl):
              if(nF<kmaxFlow):
                  HV.append(keyF+1)
          if(len(HN)==0):
              HN=None
          if(len(HL)==0):
              HL=None
          nT=True
          # if(len(NV)==0):
          #     NV=None
          #     nT=False
          # print(QN)
          # print("=======")
          # print(Vn)
          # sys.exit(1)
          #generate Image Berdasarkan kontaminasi
          # menampilkan gradient + nilai kontaminasi
          G.plot(
              # nodesindex=G.getNodeIndex(),
              # nodesID=G.getNodeIndex(),
              # nodesindex=G.getNodeIndex(),
              node_values=QN,
              link_values=QL,
              # nodesID=True,
              fig_size=[3,3],
              fontsize=3,
              # highlightnode=HN,
              # highlightlink=HL,
              figure=False,
              title=f'Persebaran Source Contaminant Node:{vln} Waktu Ke : {namelabel}',
              # colors='coolwarm',
              node_text=True,
              link_text=True,
              # colorbar='coolwarm',
              min_colorbar=0,
              max_colorbar=2,
          )
          #menyimpan Gambar ke storage
          G.plot_save(f'{directory}/resultA_{filename}_contaminat.png')
          G.plot_close()
          #membuat Plot Network yang terdapat tanda arah arus air
          netwx = nx.DiGraph()
          for key,value in enumerate(G.getNodeIndex()):
            cord=G.getNodeCoordinates(value)
            pos=(cord['x'][value],cord['y'][value])
            netwx.add_node(f'{value}', pos=pos)

          for keyLink,valueLink in enumerate(G.getLinkIndex()):
            # flow=G.getLinkFlows(valueLink)
            flow=Vl[keyLink]
            link=G.getNodesConnectingLinksID(valueLink)[0]
            if flow > 0:
                netwx.add_edge(link[0], link[1], flow=flow)
            else:
                netwx.add_edge(link[1], link[0], flow=flow)

          nodepos=nx.get_node_attributes(netwx, 'pos')
          nSize=15
          fSize=4
          plt.figure(figsize=(3, 3))
          nx.draw_networkx_nodes(netwx, nodepos, node_size=nSize, node_color='lightblue')
          nx.draw_networkx_edges(netwx, nodepos, edgelist=netwx.edges,node_size=nSize,arrowstyle='-|>',arrowsize=4, edge_color='blue')
          nx.draw_networkx_labels(netwx, nodepos, font_size=fSize, font_color='k')
          plt.title(f'Arah Arus air \npada Waktu Ke : {namelabel}',fontsize=4)
          plt.box(None)
          plt.savefig(f'{directory}/resultB_{filename}_contaminat.png')
          plt.close()
          #generate Image Berdasarkan kontaminasi berdasarkan batas aman zat
          # menampilkan ID node + nilai kontaminasi yang di atas batas aman zat
          # G.plot(
          #     # nodesindex=G.getNodeIndex(),
          #     # nodesID=G.getNodeIndex(),
          #     # nodesindex=G.getNodeIndex(),
          #     # node_values=QN,
          #     nodesID=True,
          #     fig_size=[3,3],
          #     fontsize=3,
          #     highlightnode=HN,
          #     highlightlink=HL,
          #     figure=False,
          #     title=f'Persebaran Kontaminasi Jam Ke : {namelabel} berdasarkan batas aman Chlorin= {kmax}',
          #     # colors='coolwarm',
          #     # node_text=nT,
          #     # colorbar='coolwarm',
          #     # min_colorbar=0,
          #     # max_colorbar=15,
          # )
          # #menyimpan Gambar ke storage
          # G.plot_save(f'{directory}/resultB_{filename}_contaminat.png')
          # G.plot_close()
          #generate Image Berdasarkan Besaran Flow pada Link
          # menampilkan gradient + nilai Flow
          # G.plot(
          #     link_values=Vl,
          #     fig_size=[3,3],
          #     fontsize=5,
          #     figure=False,
          #     title=f'Besaran Flow Jam Ke : {namelabel}',
          #     link_text=True,
          # )
          # #menyimpan Gambar ke storage
          # G.plot_save(f'{directory}/resultA_{filename}_flow.png')
          # G.plot_close()
          # #generate Image Berdasarkan Flow yang mengalami negative Flow
          # # menampilkan Link ID + yang mengalami negative flow
          # G.plot(
          #     nodesID=True,
          #     fig_size=[3,3],
          #     fontsize=3,
          #     figure=False,
          #     # highlightnode=HV,
          #     highlightlink=HV,
          #     title=f'Besaran Flow Jam Ke : {namelabel} berdasarkan Negative Flow',
          #     min_colorbar=0,
          #     max_colorbar=15,
          # )
          # #menyimpan Gambar ke storage
          # G.plot_save(f'{directory}/resultB_{filename}_flow.png')
          # G.plot_close()
          #menggabungkan 2 image kontaminant menjadi 1 image kontaminant dan menghapus 2 image awal
          listImage=[f'{directory}/resultA_{filename}_contaminat.png',
                    f'{directory}/resultB_{filename}_contaminat.png']
          images = [Image.open(x) for x in listImage]
          widths, heights = zip(*(i.size for i in images))
          total_width = sum(widths)
          max_height = max(heights)
          new_im = Image.new('RGB', (total_width, max_height))
          x_offset = 0
          for im in images:
              new_im.paste(im, (x_offset,0))
              x_offset += im.size[0]

          new_im.save(f'{directory}/result_{filename}_contaminat.png')
          os.remove(f'{directory}/resultA_{filename}_contaminat.png')
          os.remove(f'{directory}/resultB_{filename}_contaminat.png')

          #menggabungkan 2 image Flow menjadi 1 image Flow dan menghapus 2 image awal
          # listImageFlow=[f'{directory}/resultA_{filename}_flow.png',
          #            f'{directory}/resultB_{filename}_flow.png']
          # imagesFlow = [Image.open(x) for x in listImageFlow]
          # widths, heights = zip(*(i.size for i in imagesFlow))
          # total_width = sum(widths)
          # max_height = max(heights)
          # new_imFlow = Image.new('RGB', (total_width, max_height))
          # x_offset = 0
          # for im in imagesFlow:
          #     new_imFlow.paste(im, (x_offset,0))
          #     x_offset += im.size[0]

          # new_imFlow.save(f'{directory}/result_{filename}_flow.png')
          # os.remove(f'{directory}/resultA_{filename}_flow.png')
          # os.remove(f'{directory}/resultB_{filename}_flow.png')
      #memanggil Analisa berikutnya (agar looping bisa berjalan)
      tstep=G.nextHydraulicAnalysisStep()
      qtstep = G.nextQualityAnalysisStep()
  #menutup proses analisa
  G.closeHydraulicAnalysis()
  G.closeQualityAnalysis()
  # hr = 10

#menjalankan simulasi ulang untuk mendapat kan report (krn proses yang di atas tidak melakukan proses report)
statisticsType = 'NONE'
G.setTimeStatisticsType(statisticsType)
G.setReport('LINKS ALL')
G.setReport('NODES ALL')
# # G.setReportStatus('full')
P=G.runsCompleteSimulation()