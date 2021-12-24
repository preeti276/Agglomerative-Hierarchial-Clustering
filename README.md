# Data Analysis 
This repository contains an implementation of data analysis for both supervised and unsupervised learning.

## Unsupervised Learning: Agglomerative-Hierarchial-Clustering
Agglomerative Hierarchical Clustering is implemented to enable data analysis on unsupervised data. The dataset used is NCI microarray which has 64 columns and 6830 rows, where each column is an observation (a cell line), and each row represents a feature (a gene). Therefore, the dataset is represented via its transposed data matrix with 64 rows where each row is an observation and each observation is 6830-dimensional.The main target for data analysis on this unlabelled dataset is to understand the structure of the data by forming clusters.

#### Input: 
A dataset with 64 6830-dimensional observations. Choose a linkage (single linkage, complete linkage, average linkage and centroid linkage)

#### Algorithm:
1. Each observation forms a cluster of size 1. Thus, there are 64 clusters at
the beginning.
<br>Then repeat the following two steps, until there is only one cluster left.
2. For any pair of remaining clusters, compute the distance between them
using the linkage you chose.
3. Merge the two clusters that achieve the minimum distance into one
cluster. The distance between them will become the height of their
parent in the dendrogram.

#### Output:
A dendrogram with heirarchial clusters

The code above further implements how this dendrogram/tree can be cut at an appropriate height to get a certain number of clusters. All the code related details are added as comments and the relevant results and data analysis are added in report.
