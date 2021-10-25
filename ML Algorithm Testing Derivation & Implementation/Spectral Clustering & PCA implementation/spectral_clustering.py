import pandas as pd
import numpy as np
import networkx as nx
import sys
import warnings
from sklearn.cluster import KMeans

if not sys.warnoptions:
    warnings.simplefilter("ignore")
nodes = pd.read_csv("nodes.txt", sep="\t", header = None)
edges = pd.read_csv("edges.txt", sep="\t", header = None)
# get the list of both origin and destination
lst_edges = list(set(zip(*map(edges.get, edges))))
Graph = nx.DiGraph()
Graph.add_nodes_from(nodes[0:])
Graph.add_edges_from(lst_edges)
# remove isolated nodes
Graph.remove_nodes_from(list(nx.isolates(Graph)))
# sort the nodes
Graph_sort = sorted(Graph.nodes())
# create adjacency matrix using the graph
X = (nx.adjacency_matrix(Graph) + np.transpose(nx.adjacency_matrix(Graph)))/2
# calculate the diagonal matrix
D =  np.diag(np.array(np.sum(X, axis=1)).ravel())
L = D @ X @ D
# calculate the eigenvalues and eigenvectors of Gph_srt
eig_values, eig_vectors = np.linalg.eigh(L)
eig_vectors[np.isnan(eig_vectors)] = 0
# run kmeans
kmeans = KMeans(n_clusters=2).fit(eig_vectors)
# calculate the false classification rate
true_classification = false_classification = 0
for y,count in enumerate(Graph_sort):
    if nodes[[0,2]].loc[np.asarray((np.where(nodes[[0,2]][0]==count)))[0][0],2] == kmeans.labels_[y]:
        true_classification += 1
    else:
        false_classification+= 1
false_classification_rate = false_classification/(true_classification+false_classification)
print ("False Classification Rate:", false_classification_rate)