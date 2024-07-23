from epyt import epanet
from PIL import Image
import matplotlib.pyplot as plt
import os
import networkx as nx


networks=['BWSN-clean.inp']
# networks=['FOS - unvertices.inp','BWSN-clean.inp','Jilin including water quality.inp']
# NodeContanminants=[130]
# NodeContanminants=[37,28]
nameNetwork=['BWSN_Arrow']
# nameNetwork=['FOS_Arrow','BWSN_Arrow','JILLIN_Arrow']
for i,network in enumerate(networks):
    G=epanet(f'source_inp/data_network/{network}')
    G.plot_close()
    
    G.setQualityType('chem', 'Chlorine', 'mg/L')
    simulationDuration = 60    
    G.setTimeSimulationDuration(simulationDuration)
    patternStart = 0 
    G.setTimePatternStart(patternStart)
    patternStep = 1 
    G.setTimePatternStep(patternStep)
    G.setTimeReportingStart(patternStart)
    G.setTimeReportingStep(patternStep)
    
    directory=f"source_inp/output_simulation/plot_network/{nameNetwork[i]}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # nodeIndex=NodeContanminants[i]
    nodeIndex=G.getNodeIndex()[-1]
    sourceType = 'SETPOINT' #MASS,CONCEN,SETPOINT,FLOWPACED
    G.setNodeSourceType(nodeIndex, sourceType)
    
    s=[]
    nodeIndexA=[nodeIndex] 
    sourceStrength=1.5
    for ik,vk in enumerate(G.getNodeIndex()):
        if vk in nodeIndexA:
            s.append(sourceStrength)
        else:
            s.append(0.0)
    G.setNodeInitialQuality(G.getNodeIndex(),s)
    G.setNodeSourceQuality(G.getNodeIndex(),s)
    
    G.openHydraulicAnalysis()
    G.openQualityAnalysis()
    G.initializeHydraulicAnalysis(0)
    G.initializeQualityAnalysis(G.ToolkitConstants.EN_NOSAVE)

    plotReport=1 
    plotStepTimer=plotReport*60*patternStep
    plotTimer=1*60*patternStep

    tstep, P,T = 1, [],[]
    while tstep>0:
      t = G.runHydraulicAnalysis()
      qt = G.runQualityAnalysis()
      if int(t)%plotStepTimer== 0 :
        Vl=G.getLinkFlows() 
        filename=int(t/plotTimer) 
        G.plot(
            # linksID=True,
            # nodesID=True,
            fig_size=[20,20],
            fontsize=3,
            figure=False,
            # title=f'NetWork :{network}'
        )
        labelTest=[]
        posN={}
        for key,value in enumerate(G.getNodeIndex()):
            cord=G.getNodeCoordinates(value)
            nameNode=G.getNodeNameID(value)
            posN[int(nameNode)]=[cord['x'][value],cord['y'][value]]
        for keyLink,valueLink in enumerate(G.getLinkIndex()):
            flow=Vl[keyLink]
            link=G.getNodesConnectingLinksID(valueLink)[0]
            if flow > 0:
                labelTest.append([int(link[0]), int(link[1]),str(valueLink)])
            else:
                labelTest.append([int(link[1]),int(link[0]),str(valueLink)])
        
        for n1, n2, label in labelTest:
            x = (posN[n1][0] + posN[n2][0]) / 2
            y = (posN[n1][1] + posN[n2][1]) / 2
            plt.text(x, y, label, fontsize=4, color='red', bbox=dict(facecolor='white', edgecolor='none', alpha=0.1))
            plt.annotate('', xy=(posN[n2][0], posN[n2][1]), xytext=(posN[n1][0], posN[n1][1]),arrowprops=dict(arrowstyle='->', lw=1, color='blue'))
        plt.title(f'Arah Arus air Network: {network}',fontsize=40)
        G.plot_save(f'{directory}/plotA_{filename}.png')
        G.plot_close()
        # netwx = nx.DiGraph()
        # for key,value in enumerate(G.getNodeIndex()):
        #     cord=G.getNodeCoordinates(value)
        #     pos=(cord['x'][value],cord['y'][value])
        #     netwx.add_node(f'{value}', pos=pos)
        # for keyLink,valueLink in enumerate(G.getLinkIndex()):
        #     flow=Vl[keyLink]
        #     link=G.getNodesConnectingLinksID(valueLink)[0]
        #     if flow > 0:
        #         netwx.add_edge(str(link[0]), str(link[1]), flow=flow,name=str(valueLink),pos=('1','1'))
        #     else:
        #         netwx.add_edge(str(link[1]), str(link[0]), flow=flow,name=str(valueLink),pos=('1','1'))
        
        # nodepos=nx.get_node_attributes(netwx, 'pos')
        # edge_labels = nx.get_edge_attributes(netwx, 'name')
        # frameSize=30#15
        # nSize=100#100
        # fSize=14#14
        # node_list=None
        # edge_list=netwx.edges
        # node_listdict=None
        # edge_labelList=edge_labels.items()
        # if nameNetwork[i]=='BWSN_Arrow':
        #     ignoreNode=['0','130','131']
        #     # fSize=30
        #     edge_labelList={((k, v),labeledge) for (k, v),labeledge in edge_labels.items() if k not in ignoreNode and v not in ignoreNode}
        #     node_list=[str(v) for v in range(1,len(nodepos)+1)]
        #     node_listdict={str(v):str(v) for v in range(1,len(nodepos)+1)}
        #     edge_list=[tup for tup in netwx.edges if '0' not in tup and '130' not in tup and '131' not in tup]
        # plt.figure(figsize=(frameSize,frameSize))
        # nx.draw_networkx_nodes(netwx, nodepos,nodelist=node_list, node_size=nSize+2, node_color='lightblue') #nSize+200
        
        # nx.draw_networkx_edges(netwx, nodepos, edgelist=edge_list,node_size=nSize,arrowstyle='-|>',arrowsize=fSize+1, edge_color='blue')#fSize+18
        # nx.draw_networkx_labels(netwx, nodepos,labels=node_listdict, font_size=fSize, font_color='k')
        # # nx.draw_networkx_edge_labels(netwx, nodepos, edge_labels=edge_labels, font_size=fSize, verticalalignment="bottom",)
        
        # for (n1, n2), label in edge_labelList:
        #     x = (nodepos[n1][0] + nodepos[n2][0]) / 2
        #     y = (nodepos[n1][1] + nodepos[n2][1]) / 2
        #     plt.text(x, y, label, fontsize=fSize+2, color='red', bbox=dict(facecolor='white', edgecolor='none', alpha=0.1))

        # plt.title(f'Arah Arus air Network: {network}',fontsize=fSize+20)
        # plt.box(None)
        # plt.savefig(f'{directory}/plot_{filename}.png')
        # plt.close()

        # listImage=[f'{directory}/plotA_{filename}.png',
        #             f'{directory}/plot_{filename}.png']
        # images = [Image.open(x) for x in listImage]
        # widths, heights = zip(*(i.size for i in images))
        # total_width = sum(widths)
        # max_height = max(heights)
        # new_im = Image.new('RGB', (total_width, max_height))
        # x_offset = 0
        # for im in images:
        #     new_im.paste(im, (x_offset,0))
        #     x_offset += im.size[0]

        # new_im.save(f'{directory}/plot_{filename}.png')
        # os.remove(f'{directory}/plotA_{filename}.png')
        # # os.remove(f'{directory}/plotB_{filename}.png')

      tstep=G.nextHydraulicAnalysisStep()
      qtstep = G.nextQualityAnalysisStep()
    
    G.closeHydraulicAnalysis()
    G.closeQualityAnalysis()