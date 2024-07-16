from epyt import epanet
# from PIL import Image
import matplotlib.pyplot as plt
import os
import networkx as nx


networks=['FOS - unvertices.inp']
NodeContanminants=[37]
nameNetwork=['FOS']
for i,network in enumerate(networks):
    G=epanet(f'source_inp/data_network/{network}')
    G.plot_close()
    
    G.setQualityType('chem', 'Chlorine', 'mg/L')
    simulationDuration = 360    
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

    nodeIndex=NodeContanminants[i]
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
        # G.plot(
        #     linksID=True,
        #     nodesID=True,
        #     fig_size=[3,3],
        #     fontsize=3,
        #     figure=False,
        #     title=f'NetWork :{network}'
        # )
        # G.plot_save(f'{directory}/plotA_{filename}.png')
        # G.plot_close()
        netwx = nx.DiGraph()
        for key,value in enumerate(G.getNodeIndex()):
            cord=G.getNodeCoordinates(value)
            pos=(cord['x'][value],cord['y'][value])
            netwx.add_node(f'{value}', pos=pos)
        for keyLink,valueLink in enumerate(G.getLinkIndex()):
            flow=Vl[keyLink]
            link=G.getNodesConnectingLinksID(valueLink)[0]
            if flow > 0:
                netwx.add_edge(link[0], link[1], flow=flow,name=valueLink)
            else:
                netwx.add_edge(link[1], link[0], flow=flow,name=valueLink)

        nodepos=nx.get_node_attributes(netwx, 'pos')
        edge_labels = nx.get_edge_attributes(netwx, 'name')
        plt.figure(figsize=(15,15))
        nSize=100
        fSize=14
        nx.draw_networkx_nodes(netwx, nodepos, node_size=nSize, node_color='lightblue')
        nx.draw_networkx_edges(netwx, nodepos, edgelist=netwx.edges,node_size=nSize,arrowstyle='-|>',arrowsize=fSize+6, edge_color='blue')
        nx.draw_networkx_labels(netwx, nodepos, font_size=fSize-2, font_color='k')
        # nx.draw_networkx_edge_labels(netwx, nodepos, edge_labels=edge_labels, font_size=fSize, verticalalignment="bottom",)
        
        for (n1, n2), label in edge_labels.items():
            x = (nodepos[n1][0] + nodepos[n2][0]) / 2
            y = (nodepos[n1][1] + nodepos[n2][1]) / 2
            plt.text(x, y, label, fontsize=fSize, color='red', bbox=dict(facecolor='white', edgecolor='none', alpha=0.1))

        plt.title(f'Arah Arus air ',fontsize=fSize+2)
        plt.box(None)
        plt.savefig(f'{directory}/plot_{filename}.png')
        plt.close()

        # listImage=[f'{directory}/plotA_{filename}.png',
        #             f'{directory}/plotB_{filename}.png']
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
        # os.remove(f'{directory}/plotB_{filename}.png')

      tstep=G.nextHydraulicAnalysisStep()
      qtstep = G.nextQualityAnalysisStep()
    
    G.closeHydraulicAnalysis()
    G.closeQualityAnalysis()