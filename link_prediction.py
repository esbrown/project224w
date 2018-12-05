import math
import numpy as np
import pandas as pd
import random
import networkx as nx
from matplotlib import pyplot as plt
import csv
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix
from collections import Counter

def buildGraph():
    df_0 = pd.read_csv('files/0.txt', sep='\t', header=None)
    df_1 = pd.read_csv('files/1.txt', sep='\t', header=None)
    df_2 = pd.read_csv('files/2.txt', sep='\t', header=None)
    df_3 = pd.read_csv('files/3.txt', sep='\t', header=None)
    df = pd.concat([df_0, df_1, df_2, df_3])

    G = nx.DiGraph()
    views = {}
    print "Adding nodes to Graph"
    edgeList = open('youtube.edgelist', 'w') 
    for _, row in df.iterrows():
        if not row[3] == 'Comedy':
            continue
        nodeID = row[0]
        views[nodeID] = row[5] if row[5] == 0 else math.log(row[5], 10)
        if not G.has_node(nodeID):
            G.add_node(nodeID)

    newToOld = {}
    oldToNew = {}
    newIDIter = 0
    for _, row in df.iterrows():
        if not row[3] == 'Comedy':
            continue
        edges = row[9:]
        nodeID = row[0]
        if nodeID not in oldToNew:
            oldToNew[nodeID] = newIDIter
            newToOld[newIDIter] = nodeID
            newIDIter += 1
        for edge in edges:
            if not G.has_node(edge):
                continue
            G.add_edge(nodeID, edge)
            if edge not in oldToNew:
                oldToNew[edge] = newIDIter
                newToOld[newIDIter] = edge
                newIDIter += 1
            edgeList.write(str(oldToNew[nodeID]) + ' ' + str(oldToNew[edge]) + '\n')

    print "Graph loaded"
    print str(nx.number_of_nodes(G)) + " nodes!!"
    edgeList.close()
    np.save("newToOldMap.npy", newToOld)
    return G, views


def mapIDsToRepresentations():
    newToOld = np.load('newToOldMap.npy').item()
    path = 'youtube.emd'
    node2rep = {}
    allNodeIDs = []
    with open(path, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            vec = row[0].split()
            temp_node_id = int(vec[0])
            nodeID = newToOld[temp_node_id]
            allNodeIDs.append(nodeID)
            rep = [float(feature) for feature in vec[1:]]
            node2rep[nodeID] = rep
    return node2rep, allNodeIDs

def buildDatasets(G, node2rep, allNodeIDs):
    allEdges = list(G.edges())
    edgesSet = set(allEdges)
    numEdges = len(allEdges)
    goalPositiveExamples = 0.5*numEdges
    goalNegativeExamples = 0.5*numEdges
    numPositiveExamples = 0
    numNegativeExamples = 0
    datasets = [[] for i in range(4)]
    nodePairsInDataset = set()
    while numNegativeExamples < goalNegativeExamples:
        node1 = random.choice(allNodeIDs)
        node2 = random.choice(allNodeIDs)
        if node1 == node2:
            continue
        potentialEdge = node1, node2
        if potentialEdge not in edgesSet and potentialEdge not in nodePairsInDataset:
            rep1 = node2rep[node1]
            rep2 = node2rep[node2]
            concat = rep1 + rep2
            hadamard = [a*b for a,b in zip(rep1, rep2)]
            sums = [a+b for a,b in zip(rep1, rep2)]
            avg = [(a+b)/2 for a,b in zip(rep1, rep2)]
            if len(concat) >= 256:
                datasets[0].append([concat, 0])
                datasets[1].append([hadamard, 0])
                datasets[2].append([sums, 0])
                datasets[3].append([avg, 0])
                nodePairsInDataset.add(potentialEdge)
                numNegativeExamples += 1
    random.shuffle(allEdges)
    for i in range(int(goalPositiveExamples)):
        edge = allEdges[i]
        rep1 = node2rep[edge[0]]
        rep2 = node2rep[edge[1]]
        concat = rep1 + rep2
        hadamard = [a*b for a,b in zip(rep1, rep2)]
        sums = [a+b for a,b in zip(rep1, rep2)]
        avg = [(a+b)/2 for a,b in zip(rep1, rep2)]
        if len(concat) >= 256:
            datasets[0].append([concat, 1])
            datasets[1].append([hadamard, 1])
            datasets[2].append([sums, 1])
            datasets[3].append([avg, 1])
    return datasets

def splitData(data):
    X = np.array([entry[0] for entry in data])
    Y = np.array([entry[1] for entry in data])
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
    return trainX, testX, trainY, testY

def logisticRegression(trainX, testX, trainY, testY):
    logisticRegr = LogisticRegression(verbose=True)
    logisticRegr.fit(trainX, trainY)
    predictions = logisticRegr.predict(testX)
    score = logisticRegr.score(testX, testY)
    cm = metrics.confusion_matrix(testY, predictions)
    print 'cm: ', cm
    print 'score: ', score

def neuralNet(trainX, testX, trainY, testY):
    mlp = MLPClassifier(hidden_layer_sizes=(15, 15, 5),max_iter=500)
    mlp.fit(trainX, trainY)
    predictions = mlp.predict(testX)
    print(confusion_matrix(testY,predictions))
    print(classification_report(testY,predictions))

def plotHubsVsViews(G):
    hubs, authorities = nx.hits(G, max_iter = 10000, tol=1.0e-10)
    print "Finished Hits algorithm"

    sortedHubs = sorted(hubs.items(), key=lambda tuple: tuple[1], reverse=True)
    sortedViews = sorted(views, key=views.get, reverse=True)

    top100Views = sortedViews[0:100]
    print top100Views
    for i in range(25):
        hub = sortedHubs[i][0]
        count = 0
        print G.out_edges(hub)
        for edge in G.out_edges(hub):
            if edge[1] in top100Views:
                count += 1
        print "hub: " + str(hub) + " top 100: " + str(count)

    x, y = [views[i] for i in views], [hubs[j] for j in views]
    plt.xlabel('Views (log)')
    plt.ylabel('Hub Ranking')
    plt.title('Hub Ranking vs View Count For Comedy Videos')
    config = plt.gca()
    config.scatter(x,y)
    plt.show()

def jacaard(G):
    jaccard = nx.jaccard_coefficient(G)
    averages = []
    print len(removedEdges), 'total'
    numPrinted = 0
    for u, v, p in jaccard:
        print u,v,p
        if (u,v) in removedEdges:
            if numPrinted % 100 == 0:
                print numPrinted
            averages.append(p)
            numPrinted += 1
    print averages
    print 'total average', sum(averages)/len(averages)
    pyplot.hist(averages, bins=10, color='c')
    pyplot.xlabel('Jacaard Coefficient of Removed Edge Pairs')
    pyplot.ylabel('Number of Edge Pairs')
    pyplot.show()


if __name__ == "__main__":
    print "Loading Graph in Network X!"
    G, views = buildGraph()
    jacaard(G)
    # plotHubsVsViews(G)
    
    # node2rep, allNodeIDs = mapIDsToRepresentations()
    # datasetName = {0: 'concat', 1: 'hadamard', 2: 'sum', 3: 'avg'}
    # for seed in [1,2,3,4,5]:
    #     random.seed(seed)
    #     print '__________________seed', seed, '___________________'
    #     datasets = buildDatasets(G, node2rep, allNodeIDs)
    #     for i, dataset in enumerate(datasets):
    #         print '____________', datasetName[i], '____________'
    #         trainX, testX, trainY, testY = splitData(dataset)
    #         neuralNet(trainX, testX, trainY, testY)
    #     print ''

    # for seed in [1,2,3,4,5]:
    #     random.seed(seed)
    #     print '__________________seed', seed, '___________________'
    #     datasets = buildDatasets(G, node2rep, allNodeIDs)
    #     for i, dataset in enumerate(datasets):
    #         print '____________', datasetName[i], '____________'
    #         trainX, testX, trainY, testY = splitData(dataset)
    #         logisticRegression(trainX, testX, trainY, testY)
    #     print ''


