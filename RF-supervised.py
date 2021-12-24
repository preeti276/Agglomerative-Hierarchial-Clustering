import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def avg(data,indx):
    if len(indx) == 0:	return 0.0
    return sum([ data[i][-1] for i in indx ]) / len(indx)

def rss(data,indx):
    if len(indx) == 0:	return 0.0
    mean = avg(data,indx)
    return sum([ pow( data[i][-1]-mean , 2.0 ) for i in indx ])
  
#Creating Decision Tree class which will have optimal split at each node 
#using recursive splitting  
class decisionTree:
    	
    def __init__(self,data,indx,depth):
        
        if depth==0 or len(indx)==0:
            self.leaf = True
            self.prediction = avg(data,indx)
        elif len( set([data[i][-1] for i in indx]) ) == 1:
            self.leaf = True
            self.prediction = data[indx[0]][-1]
        else:
            self.leaf = False
            self.attr , self.split , self.L , self.R = self.generate(data,indx,depth)
                      
    def generate(self,data,indx,depth):
        p = len(data[indx[0]])-1
        labels = [ data[i][-1] for i in indx ]
        feature_subset = random.sample(range(p), k=math.ceil(p/3))
        opt = pow ( max(labels) - min(labels) , 2.0 ) * len(indx) + 1.0
        for feature_index in feature_subset:
            all_cuts = set([ data[i][feature_index] for i in indx ])
            for cut in all_cuts:
                yl = [ i for i in indx if data[i][feature_index]<=cut ]
                yr = [ i for i in indx if data[i][feature_index]>cut ]
                tmp = rss(data,yl) + rss(data,yr)
                if tmp < opt:
                    opt , attr , split, L , R = tmp , feature_index , cut , yl , yr
        return attr , split , decisionTree(data,L,depth-1) , decisionTree(data,R,depth-1)
    
    def predict(self,x):
        if self.leaf == True:	
            return self.prediction
        if (x[self.attr] <= self.split):	
            return self.L.predict(x)
        return self.R.predict(x)
    
##################################RANDOM FOREST###################################
    
#Loading the dataset
boston = datasets.load_boston()

#Creating our data from dataset
data = pd.DataFrame()
for i in range(len(boston['feature_names'])):
        data[str(boston['feature_names'][i])] = boston.data[:,i]        
data['MEDV'] = boston.target

#Factorising the categorical/qualitative variables
data['CHAS'] = data['CHAS'].astype('category')

X = data[data.columns[:-1]] #Features
y = data[data.columns[-1]] #targets

#Splitting the dataset randomly into two equal test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=520) 

'''
(a) Generate B = 100 bootstrapped training sets (BTS) from the training set.
'''
#Function to generate B BTS of height h each from the training data
def getBDecisionTrees(B,h,train_data):
    sample_count = train_data.shape[0]
    random.seed(520)
    hatf = []	#this is used to store the decision trees generated
    
    for i in range(B):
        #Generating BTS
        b_dataset = train_data.sample(sample_count, replace=True)
        b_dataset = b_dataset.drop_duplicates()
        b_dataset = b_dataset.sort_index()
        #making decision tree for each BTS
        tree_data = np.array(b_dataset)
        data_length = len(tree_data)
        all_indx = list(range(data_length))
        tree = decisionTree(tree_data,all_indx,h)
        #storing our decision tree in list of trees
        hatf.append(tree)
    return hatf

'''
b) Use each BTS to train for a decision tree of height h = 3.
'''
    
B = 100
height = 3
train_data = pd.DataFrame(data=X_train)
train_data[data.columns[-1]] = y_train
hatf = getBDecisionTrees(B,height,train_data)

'''
c) Find the training MSE and test MSE.
'''
#Generic function to find MSE given actual labels and decision trees alongwith data
def getMSE(hatf, data, labels):  
    
    #Getting predictions for training data from our BTS
    yhats = []
    for i in range(len(data)):
        X = data.iloc[[i]]    
        to_predict = np.array(X)[0]
        yhat = np.mean([hatf[j].predict(to_predict[:-1]) for j in range(len(hatf)) ])
        yhats.append(yhat)
           
    yhats = np.array(yhats)
    y = np.array(labels)
    diff = []
    for i in range(len(y)):
        true_value = y[i]
        pred_value = yhats[i]
        diff.append((true_value - pred_value)**2)
    mse = sum(diff)/len(diff)
    return mse

print('RESULT: B = 100, h = 3')

#Finding training MSE
train_mse = getMSE(hatf, train_data, y_train)
print('Training MSE:',train_mse)

#Finding Testing MSE
test_data = pd.DataFrame(data=X_test)
test_data[data.columns[-1]] = y_test
test_mse = getMSE(hatf, test_data, y_test)
print('Test MSE:',test_mse)

'''
d)Repeat the above parts using different values of B and h. In your report,
plot the training MSE and test MSE as functions of B or/and h, and
discuss your observations.
'''

#The below code is written to produce results as presented in report
#(can be commented)

#Repeating the above parts for different values of B and height=3
B = [1, 5, 10,20, 25, 30, 40 ,50, 100]
B_train_mses = []
B_test_mses = []
for b in B:
    fhat = getBDecisionTrees(b,3,train_data)
    B_train_mses.append(getMSE(fhat, train_data, y_train))
    B_test_mses.append(getMSE(fhat, test_data, y_test))
    
plt.plot(B, B_train_mses, 'r', label = "Train MSE")
plt.plot(B, B_test_mses, 'b', label = "Test MSE")
plt.legend()
plt.show()

print('b_train_mses',B_train_mses)
print('b_test_mses',B_test_mses)

#Repeating the above parts for different values of h and B = 100
h = [1, 2, 3, 4, 5, 6, 8, 10]
h_train_mses = []
h_test_mses = []
for height in h:
    fhat = getBDecisionTrees(100,height,train_data)
    h_train_mses.append(getMSE(fhat, train_data, y_train))
    h_test_mses.append(getMSE(fhat, test_data, y_test))
    
print('h_train_mses',h_train_mses)
print('h_test_mses',h_test_mses)

plt.plot(h, h_train_mses, 'r', label = "Train MSE")
plt.plot(h, h_test_mses, 'b', label = "Test MSE")
plt.legend()
plt.show()