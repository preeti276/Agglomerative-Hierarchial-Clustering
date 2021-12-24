# Data Analysis 
This repository contains an implementation of data analysis for both supervised and unsupervised learning.

## Supervised Learning: Random Forest
Random Forest, an ensemble method, is implemented to enable data analysis on supervised data. It is implemented on the Boston dataset where medv is the label, and the other 13 variables in the dataset are the attributes. Since the label here is not categorical, it is a regression problem for data analysis. We will use 100 decision trees, each of height 3, to make an effective random forest model.

#### Input: 
Boston dataset with 506 13-attribute records and a label field for each record. 

#### Algorithm:
Random forest method is similar to using recursive binary splitting ,i.e. generating a decision tree with the optimal attribute and split value at each node, except:

1. instead of generating only one decision tree, now you generate B decision trees, while each decision tree is generated using one BTS;
2. instead of considering all attributes at each node, now you consider only √p random attributes at each node in regression and p/3 attributes for classification problem.
3. the prediction is made by a majority vote among the predictions from the B decision trees;

#### Output:
Analyze the data using the error rate from all B decision trees.

All the code related details are added in code file. The effect of changing the number of decision trees considered and the height of each tree is presented in attached report.

## Unsupervised Learning: Agglomerative Hierarchial Clustering
Agglomerative Hierarchical Clustering is implemented to enable data analysis on unsupervised data. The dataset used is NCI microarray which has 64 columns and 6830 rows, where each column is an observation (a cell line), and each row represents a feature (a gene). Therefore, the dataset is represented via its transposed data matrix with 64 rows where each row is an observation and each observation is 6830-dimensional.The main target for data analysis on this unlabelled dataset is to understand the structure of the data by forming clusters.

#### Input: 
NCI dataset with 64 6830-dimensional observations. Choose a linkage (single linkage, complete linkage, average linkage and centroid linkage)

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
