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

def plotStats(title, precisions, recalls, F1s):
	# create plot
	fig, ax = plt.subplots()
	index = np.arange(0.2, n_groups)
	bar_width = 0.2
	opacity = 0.8
	 
	rects1 = plt.bar(index, precisions, bar_width,
	                 alpha=opacity,
	                 color='b',
	                 label='Precision')
	 
	rects2 = plt.bar(index + bar_width, recalls, bar_width,
	                 alpha=opacity,
	                 color='g',
	                 label='Recall')

	rects3 = plt.bar(index + 2*bar_width, F1s, bar_width,
	                 alpha=opacity,
	                 color='r',
	                 label='F1-Score')
	 
	plt.xlabel('Edge Representation')
	plt.ylabel('Score')
	plt.title('Comparing Edge Representation Methods: MLP Classifier')
	plt.xticks(index + 1.5*bar_width, ('Concatenate', 'Hadamard', 'Sum', 'Average'))
	plt.legend(loc=1, ncol=1)
	plt.ylim([.5, 1])
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	n_groups = 4

	# logistic regression data
	precisions = [0.65922304, 0.991125719, 0.663436799, 0.680004379]
	recalls = [0.670294459, 0.978105741, 0.660976275, 0.659324912]
	F1s = [0.66471265, 0.984572688, 0.662204251, 0.669504999]
	title = 'Comparing Edge Representation Methods: Logistic Regression'
	plotStats(title, precisions, recalls, F1s)

	# ML Classifier data
	precisions = [0.977623654, 0.984755097, .970491089, 0.96370269]
	recalls = [0.995576373, 0.996604289, 0.97342362, 0.980332127]
	F1s = [0.986518344, 0.990644262, 0.971955143, 0.971946284]
	title = 'Comparing Edge Representation Methods: ML Classifier'
	plotStats(title, precisions, recalls, F1s)

