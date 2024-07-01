from epyt import epanet
import csv
import os

name_network="jilin"
# name_network="Jilin including water quality"
# G = epanet('source_inp/net2-cl2.inp')
# G.loadMSXFile('source_inp/net2-cl2.msx')
G = epanet('source_inp/data_network/Jilin including water quality.inp')
# G = epanet('source_inp/network/New York Tunnels including water quality.inp')
# G = epanet(f'source_inp/data_network/FOS - unvertices.inp')
# G = epanet('source_inp/Net3-NH2CL.inp')
# G.loadMSXFile('source_inp/Net3-NH2CL.msx')

simulationDuration = 86400    #Duration-> 172800 seconds = 2days | default => 086400 = 1 days
G.setTimeSimulationDuration(simulationDuration)

patternStart = 0 # Pattern Start
G.setTimePatternStart(patternStart)
patternStep = 1 # Pattern Timestep
G.setTimePatternStep(patternStep)

nodeIndex=28 #15
sourceStrength=1.5 #besaran kontaminant yang akan di injeksi (yang merubah setingan default dari Network yang digunakan)
sourceType = 'SETPOINT' #MASS,CONCEN,SETPOINT,FLOWPACED
G.setNodeSourceType(nodeIndex, sourceType)
s=[]
for ikz,vkz in enumerate(G.getNodeIndex()):
    if vkz == nodeIndex:
        s.append(sourceStrength)
    else:
        s.append(0.0)
G.setNodeInitialQuality(G.getNodeIndex(),s)
G.setNodeSourceQuality(G.getNodeIndex(),s)
# print(G.getNodeInitialQuality(nodeIndexA))

# # reportingStart = 0 # Report Start
# # G.setTimeReportingStart(reportingStart)
# # reportingStep = 300 # Report Timestep
# # G.setTimeReportingStep(reportingStep)

# # simulationAccuracy = 0.001
# # G.setOptionsAccuracyValue(simulationAccuracy)
# # QualityTolerance=0.01
# # G.setOptionsQualityTolerance(QualityTolerance)
# # EmiterExponent=0.5
# # G.setOptionsEmitterExponent(EmiterExponent)
exit()
statisticsType = 'NONE'
G.setTimeStatisticsType(statisticsType)
directory="source_inp/output_simulation/report_simulation"
if not os.path.exists(directory):
    os.makedirs(directory)

G.setReportFormatReset()
G.setReport(f'FILE {directory}/{name_network}_node.csv')
G.setReport('NODES ALL')
G.setReport('SUMMARY NO')
# # # # Run
P=G.runsCompleteSimulation()
#----------------------------------------------------------------------------------
G.setReportFormatReset()
G.setReport(f'FILE {directory}/{name_network}_links.csv')
G.setReport('LINKS ALL')
G.setReport('SUMMARY NO')
# # # # Run
P=G.runsCompleteSimulation()
#----------------------------------------------------------------------------------
# G.setReport('STATUS YES')

# # # G.setReport('LENGTH YES')
# # # G.setReport('ELEVATION YES')
# # # G.setReport('DEMAND YES')
# # # G.setReport('HEAD YES')
# # # G.setReport('PRESSURE YES')
# # # G.setReport('QUALITY YES')
# G.setReport('LINKS ALL')
# G.setReport('NODES ALL')
# # G.setReportStatus('full')
# # # line = 'Status YES'
# # # G.writeLineInReportFile(line)
# # # G.writeReport()
# # # report_file_string = open('TestReport3.txt')
# P=G.runsCompleteSimulation()


# # #Menjalankan simulasi
# # qualInfo=G.getQualityInfo()
# # qualInfo=G.getLinksInfo()

# #-------------------------------------------------------------------
filenameNode=f'{directory}/{name_network}_node.csv'
exit()
line_to_delete = 11
initial_line = 1
file_lines = {}
labelNode=''

with open(filenameNode,"r") as f:
    content = f.readlines()
for line in content:
    file_lines[initial_line] = line.strip()
    initial_line += 1
f = open(filenameNode, "w")
f.write("node_result\tnode\tdemand_gpm\thead_ft\tpressure_psi\tQuality\n")
for line_number, line_content in file_lines.items():
    if 'Node Results' in line_content:
        labelNode=line_content[16:-5]
    if line_number >= line_to_delete and 'Node' not in line_content and 'Demand' not in line_content and 'Node Results' not in line_content and line_content!='' and line_content!='--------------------------------------------------------':
        f.write(format(labelNode)+"\t")
        f.write(format(line_content))
        f.write('\n')
f.close()
print('Deleted line: {}'.format(line_to_delete))
#------------------------------------------------------------------
filenameLink=f'{directory}/{name_network}_links.csv'
line_to_delete = 11
initial_line = 1
file_lines = {}
labelLinks=''

with open(filenameLink,"r") as f:
    content = f.readlines()
for line in content:
    file_lines[initial_line] = line.strip()
    initial_line += 1
f = open(filenameLink, "w")
f.write("links_result\tlink\tflow_gpm\tvelocity_fps\theadloss\n")
for line_number, line_content in file_lines.items():
    if 'Link Results' in line_content:
        labelLinks=line_content[16:-5]
    if line_number >= line_to_delete and 'Link' not in line_content and 'Flow' not in line_content and 'Link Results' not in line_content and line_content!='' and line_content!='----------------------------------------------':
        f.write(format(labelLinks)+"\t")
        f.write(format(line_content))
        f.write('\n')
f.close()
print('Deleted line: {}'.format(line_to_delete))
#------------------------------------------------------------------
pipes=G.getLinkPipeNameID()
with open(f'{directory}/{name_network}_pipe.csv', 'w',newline='') as csvfile:
    writerpipes = csv.writer(csvfile, dialect='excel')
    writerpipes.writerow(['pipeID','length','diameter','roughnes'])
    for index,pipe in enumerate(pipes):
        i=index+1
        diameter=G.getLinkDiameter(i)
        roughnes=G.getLinkRoughnessCoeff(i)
        length=G.getLinkLength(i)
        writerpipes.writerow([pipe,length,diameter,roughnes])
#Export PIPES to CSV
#------------------------------------------------------------------
junctionIndexs=G.getNodeJunctionIndex()
with open(f'{directory}/{name_network}_junction.csv', 'w',newline='') as csvfile:
    writerjunction = csv.writer(csvfile, dialect='excel')
    writerjunction.writerow(['junctionID','elev','demand','pattern'])
    for junction in junctionIndexs:
        junctionID=G.getNodeJunctionNameID(junction)
        jelev=G.getNodeElevations(junction)
        writerjunction.writerow([junctionID,jelev,"",""])
#Export JUNCTIONS to CSV
#------------------------------------------------------------------
reservoirIndex=G.getNodeReservoirIndex()
with open(f'{directory}/{name_network}_reservoire.csv', 'w',newline='') as csvfile:
    writerreservoir = csv.writer(csvfile, dialect='excel')
    writerreservoir.writerow(['reservoirID','head','pattern'])
    for reservoir in reservoirIndex:
        reservoirIndexID=G.getNodeNameID(reservoir)
        writerreservoir.writerow([reservoirIndexID,"",""])
# Export Reservoirs to CSV
#------------------------------------------------------------------
tankIndex=G.getNodeTankData().to_dict()
# print(tankIndex)
with open(f'{directory}/{name_network}_tank.csv', 'w',newline='') as csvfile:
    writertank = csv.writer(csvfile, dialect='excel')
    writertank.writerow(['tankID','elevation','initlevel','minlevel','maxlevel','diameter'])
    for it,reservoir in enumerate(tankIndex['Index']):
        labelTank=G.getNodeJunctionNameID(reservoir)
        elevTank=round(tankIndex['Elevation'][it],1)
        initlevel=round(tankIndex['Initial_Level'][it],1)
        minlevel=round(tankIndex['Minimum_Water_Level'][it],1)
        maxlevel=round(tankIndex['Maximum_Water_Level'][it],1)
        diameterTank=round(tankIndex['Diameter'][it],1)
        writertank.writerow([labelTank,elevTank,initlevel,minlevel,maxlevel,diameterTank])
#Export Tank to CSV
#------------------------------------------------------------------