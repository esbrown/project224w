import snap
import math
import numpy as np
import pandas as pd
import networkx as nx
import csv
from matplotlib import pyplot as plt

def loadGraph():
    df_0 = pd.read_csv('0.txt', sep='\t', header=None)
    df_1 = pd.read_csv('1.txt', sep='\t', header=None)
    df_2 = pd.read_csv('2.txt', sep='\t', header=None)
    df_3 = pd.read_csv('3.txt', sep='\t', header=None)
    df = pd.concat([df_0, df_1, df_2, df_3])

    G = nx.DiGraph()

    views = {}
    print "Adding nodes to Graph"
    for _, row in df.iterrows():
        if not row[3] == 'Comedy':
            continue
        nodeID = row[0]
        views[nodeID] = row[5] #if row[5] == 0 else math.log(row[5], 10)
        if not G.has_node(nodeID):
            G.add_node(nodeID)

    for _, row in df.iterrows():
        if not row[3] == 'Comedy':
            continue
        edges = row[9:]
        nodeID = row[0]
        for edge in edges:
            if not G.has_node(edge):
                continue
            G.add_edge(nodeID, edge)

    print "Graph loaded"
    print str(nx.number_of_nodes(G)) + " nodes!!"
    return G, views

def writeEdgeList(G):
    nx.write_edgelist(G, 'edgelist_comedy.csv', delimiter=',')

def writeViewCounts(views):
    with open('viewcounts.csv', mode='w') as viewcounts:
        writer = csv.writer(viewcounts, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for node in views:
            writer.writerow([node, views[node]])

def getScores():
    linscores = pd.read_csv('linscores.csv')
    linscore_dict = {}
    for _, row in linscores.iterrows():
        linscore_dict[row[0]] = row[1]

    leveragescores = pd.read_csv('leveragescores.csv')
    leverage_dict = {}
    for _, row in leveragescores.iterrows():
        leverage_dict[row[0]] = row[1]

    clusterscores = pd.read_csv('clusterrankscores.csv')
    cluster_dict = {}
    for _, row in clusterscores.iterrows():
        cluster_dict[row[0]] = row[1]

    return linscore_dict, leverage_dict, cluster_dict

if __name__ == "__main__":
    print "Loading Graph in Network X!"
    G, views = loadGraph()

    lin, leverage, cluster = getScores()
    hubs, authorities = nx.hits(G, max_iter = 10000, tol=1.0e-10)
    print "Finished Hits algorithm"

    sortedHubs = sorted(hubs.items(), key=lambda tuple: tuple[1], reverse=True)

    hubs50 = [k for k,v in sortedHubs[:50]]

    sortedLin = sorted(lin.items(), key=lambda tuple: tuple[1], reverse=True)
    sortedLev = sorted(leverage.items(), key=lambda tuple: tuple[1], reverse=True)
    sortedCluster = sorted(cluster.items(), key=lambda tuple: tuple[1], reverse=True)

    lin50 = [k for k,v in sortedLin[:50]]

    lev50 = [k for k,v in sortedLev[:50]]

    cluster50 = [k for k,v in sortedCluster[:50]]

    hubs50Views = []
    for hub in hubs50:
        neighbors = G.neighbors(hub)
        viewcounts = [views[neighbor] for neighbor in neighbors]
        if len(viewcounts) > 0:
            avg = float(sum(viewcounts))/len(viewcounts)
            hubs50Views.append(avg)

    lin50Views = []
    for hub in lin50:
        neighbors = G.neighbors(hub)
        viewcounts = [views[neighbor] for neighbor in neighbors]
        if len(viewcounts) > 0:
            avg = float(sum(viewcounts))/len(viewcounts)
            lin50Views.append(avg)
        else:
            lin50Views.append(0)

    lev50Views = []
    for hub in lev50:
        neighbors = G.neighbors(hub)
        viewcounts = [views[neighbor] for neighbor in neighbors]
        if len(viewcounts) > 0:
            avg = float(sum(viewcounts))/len(viewcounts)
            lev50Views.append(avg)
        else:
            lev50Views.append(0)

    cluster50Views = []
    for hub in cluster50:
        neighbors = G.neighbors(hub)
        viewcounts = [views[neighbor] for neighbor in neighbors]
        if len(viewcounts) > 0:
            avg = float(sum(viewcounts))/len(viewcounts)
            cluster50Views.append(avg)
        else:
            cluster50Views.append(0)
    fig, axs = plt.subplots(2, 2, sharey=True, sharex=True, tight_layout=True)

    axs[0,0].hist(hubs50Views, range=(0, 60000), bins=10)
    axs[0,0].title.set_text('Kleinbergs Algorithm')
    axs[0,1].hist(lin50Views, range=(0, 60000), bins=10)
    axs[0,1].title.set_text('Lin Centrality')
    axs[1,0].hist(lev50Views, range=(0, 60000), bins=10)
    axs[1,0].title.set_text('Leverage Centrality')
    axs[1,1].hist(cluster50Views, range=(0, 60000), bins=10)
    axs[1,1].title.set_text('ClusterRank Centrality')
    axs[0,0].set(ylabel='Count')
    axs[1,0].set(xlabel='Average Viewcount of Neighboring Nodes', ylabel='Count')
    axs[1,1].set(xlabel='Average Viewcount of Neighboring Nodes')

    ### For viewcounts vs hub/authorities
    x1, y1, y2 = [views[x] for x in views], [hubs[y] for y in views], [authorities[z] for z in views]

    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, tight_layout=True)

    axs[0].scatter(x1, y1)
    axs[0].title.set_text('Hubs')
    axs[1].scatter(x1, y2)
    axs[1].title.set_text('Authorities')

    plt.show()

