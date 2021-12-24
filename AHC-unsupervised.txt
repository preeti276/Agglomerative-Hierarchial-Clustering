import math
import pandas as pd
import numpy as np

'''
Implementing AHC with the following linkage functions: single linkage, complete linkage, average linkage and centroid linkage giving a data structure that represents a dendrogram.
'''

#Defining method to get euclidean distance between two points
def getEuclideanDistance(u, v):
    s = 0
    for i in range(len(u)):
        s = s + (u[i]-v[i])**2
    return s**0.5

#Defining method to get single linkage between two clusters
def getSingleLinkage(w1, w2):
    distance_mat = np.zeros(shape=(len(w1),len(w2)))
    for i in range(len(w1)):
        for j in range(len(w2)):
            distance_mat[i][j] = getEuclideanDistance(w1[i], w2[j])
    return distance_mat.min()

#Defining method to get complete linkage between two clusters
def getCompleteLinkage(w1, w2):
    distance_mat = np.zeros(shape=(len(w1),len(w2)))
    for i in range(len(w1)):
        for j in range(len(w2)):
            distance_mat[i][j] = getEuclideanDistance(w1[i], w2[j])
    return distance_mat.max()    
    
#Defining method to get average linkage between two clusters
def getAvgLinkage(w1, w2):
    distance_mat = np.zeros(shape=(len(w1),len(w2)))
    for i in range(len(w1)):
        for j in range(len(w2)):
            distance_mat[i][j] = getEuclideanDistance(w1[i], w2[j])
    return np.mean(distance_mat)

#Defining method to get centroid of a cluster
def getCentroid(w):
    d = len(w[0])
    centroid = np.zeros(d)
    for j in range(d):
        ci = 0
        for i in range(len(w)):
            ci = ci + w[i][j]
        centroid[j] = ci/len(w)
    return list(centroid)

#Defining method to get centroid linkage between two clusters
def getCentroidLinkage(w1, w2):
    c1 = getCentroid(w1)
    c2 = getCentroid(w2)
    return getEuclideanDistance(c1, c2)

#Defining method to get cluster matrix giving linkage values
#(as per linkage parameter) between every pair of clusters
def getClusterMat(clusters, linkage):
    cluster_mat = np.zeros(shape=(len(clusters),len(clusters)))
    for i in range(len(cluster_mat)):
        for j in range(len(cluster_mat)):
            if i >= j:
                cluster_mat[i][j] = math.inf
            else:
                if linkage == "average":
                    cluster_mat[i][j] = getAvgLinkage(clusters[i],clusters[j])
                elif linkage == "single":
                    cluster_mat[i][j] = getSingleLinkage(clusters[i],clusters[j])
                elif linkage == "centroid":
                    cluster_mat[i][j] = getCentroidLinkage(clusters[i],clusters[j])
                else:
                    cluster_mat[i][j] = getCompleteLinkage(clusters[i],clusters[j])   
    return cluster_mat

#Defining node class to store our clusters and create a dendrogram
class Node:
    def __init__(self, isLeaf, height, observations, left, right, branch):
        self.isLeaf = isLeaf
        self.height = height
        self.observations = observations
        self.left = left
        self.right = right
        self.branch = branch
        
    # Preorder traversal (Root -> Left ->Right) to get required node parameters
    
    #Traversing the dendrogram to get all heights of nodes
    def getHeights(self, root):
        heights = []
        if root:
            heights.append(root.height)
            heights = heights + self.getHeights(root.left)
            heights = heights + self.getHeights(root.right)
        return heights
    
    #Traversing the dendrogram to get all observations of nodes
    def getObservations(self, root):
        observations = []
        if root:
            observations.append(root.observations)
            observations = observations + self.getObservations(root.left)
            observations = observations + self.getObservations(root.right)
        return observations
    
    #Traversing the dendrogram to get all branches of nodes
    def getBranches(self, root):
        branches = []
        if root:
            branches.append(root.branch)
            branches = branches + self.getBranches(root.left)
            branches = branches + self.getBranches(root.right)
        return branches
        
#Defining function to perform AHC on input data x with given linkage function 
def performAHC(x, linkage):
    
    #Making each observation a cluster and storing in clusters
    clusters = [[list(x[i])] for i in range(len(x))]
    
    #Creating a data structure to store our nodes
    nodes = dict()
    
    #Creating leaf nodes for all clusters of 1 observations so far
    for i in range(len(clusters)):
        nodes[str(clusters[i])] = Node(True, 0, clusters[i], None, None, [0,math.inf])
        
    #Looping heirarchial clustering until we have 1 cluster of all observations
    while len(clusters) > 1:
        cluster_mat = getClusterMat(clusters, linkage)
        height = np.min(cluster_mat)
        
        #Finding the shortest distance between clusters
        min_cluster_indices = np.argwhere(cluster_mat == np.min(cluster_mat))
        
        #Always taking the first minimum value found
        min_cluster_indices = min_cluster_indices[0]
        
        #Merging the nearest clusters and appending to clusters
        merged_cluster = [] 
        for i in min_cluster_indices:
            for j in range(len(clusters[i])):
                merged_cluster.append(clusters[i][j])
                
        #Creating a new node for the merged cluster
        nodes[str(merged_cluster)] = Node(False, height, merged_cluster, 
                                    nodes[str(clusters[min_cluster_indices[0]])], 
                                    nodes[str(clusters[min_cluster_indices[1]])],[height,math.inf])
        
        #Updating branch field for left and right node
        nodes[str(clusters[min_cluster_indices[0]])].branch[1] = height
        nodes[str(clusters[min_cluster_indices[1]])].branch[1] = height
        
        #Adding the merged cluster to our list of clusters
        clusters.append(merged_cluster)
        
        #Removing the nearest two clusters from our list of clusters
        for i in min_cluster_indices:
            clusters[i] = 0
        while 0 in clusters:
            clusters.remove(0)
        
    #Returning the final cluster of all observations
    return nodes[str(merged_cluster)]

####################AGGLOMERATIVE HIERARCHIAL CLUSTERING##################

#Loading the nci dataset
ncidata = pd.read_fwf("ncidata.txt",header=None)
ncidata = ncidata.T

#Making dataset in a form that works for my AHC implemenatation
input_data = []
for i in range(ncidata.shape[0]):
    x = ncidata.iloc[[i]]
    data_row = np.array(x)[0]
    input_data.append(data_row)
    
#Below code was executed to obtain results as presented in report 

#Performing the AHC and getting Dendrogram data structure       
dendrogram_complete = performAHC(input_data, "complete")
dendrogram_single = performAHC(input_data, "single")
dendrogram_average = performAHC(input_data, "average")
dendrogram_centroid = performAHC(input_data, "centroid")

#Printing the height of root of dendrogram created for all linkage functions
print('Height of root of dendrograms created with following linkage:')
print('Complete Linkage:',dendrogram_complete.height)
print('Single Linkage:',dendrogram_single.height)
print('Average Linkage:',dendrogram_average.height)
print('Centroid Linkage:',dendrogram_centroid.height)


'''
Implementing a function getClusters that takes a dendrogram and a positive integer K as arguments, and its output is the K clusters obtained by
cutting the dendrogram at an appropriate height.
'''

def getClusters(dendrogram, k):
    
    heights = set(dendrogram.getHeights(dendrogram))
    branches = dendrogram.getBranches(dendrogram)
    nodes = dendrogram.getObservations(dendrogram)
    
    for height in heights:
        obtained_clusters = []
        for j in range(len(branches)):
            branch = branches[j]
            if height >= branch[0] and height < branch[1]:
                obtained_clusters.append(nodes[j])
        if len(obtained_clusters) == k:
            return obtained_clusters
    return []

'''
Evaluating the performance of AHC with the four different linkage functions when applied to the NCI microarray dataset. The results are presented in report.
'''

#Evaluating the Potential Function Value of a cluters assignment
def getPotentialFunction(clusters):
    res = 0
    for i in range(len(clusters)):
        cluster = clusters[i]
        centroid = getCentroid(cluster)
        value = 0
        for j in range(len(cluster)):
            observation = cluster[j]
            value = value + getEuclideanDistance(observation, centroid)
        res = res + value
    return res

#Below code was executed to obtain results as presented in report
#(can be commented out) 

k = [3,5,10,15,20,50,60]
for kval in k:
    print('k:',kval)
    
    #Evaluating the potential function value of k clusters obtained
    c1 = getClusters(dendrogram_complete, k)
    c2 = getClusters(dendrogram_single, k)
    c3 = getClusters(dendrogram_average, k)
    c4 = getClusters(dendrogram_centroid, k)
    
    print('Potential Function Values')
    print('complete:',getPotentialFunction(c1))
    print('single:',getPotentialFunction(c2))
    print('average:',getPotentialFunction(c3))
    print('centroid:',getPotentialFunction(c4))
            
    #Evaluating the minimum and maximum size of clusters obtained
    cluster_lengths1 = [len(c1[i]) for i in range(len(c1))]
    cluster_lengths2 = [len(c2[i]) for i in range(len(c2))]
    cluster_lengths3 = [len(c3[i]) for i in range(len(c3))]
    cluster_lengths4 = [len(c4[i]) for i in range(len(c4))]
       
    print('min and max values cluster sizes')
    print('complete:',max(cluster_lengths1),min(cluster_lengths1))
    print('single:',max(cluster_lengths2),min(cluster_lengths2))
    print('average:',max(cluster_lengths3),min(cluster_lengths3))
    print('centroid:',max(cluster_lengths4),min(cluster_lengths4))