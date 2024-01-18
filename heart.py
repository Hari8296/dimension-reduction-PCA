Name:-Hari singh r
Batch id:-DSWDMCOD 25082022 B

import pandas as pd  
import matplotlib.pylab as plt  
import matplotlib.pyplot  as plt  
from scipy.cluster.hierarchy import linkage,dendrogram 
from sklearn.cluster import AgglomerativeClustering 
import numpy as np


df=pd.read_csv("D:/assignments of data science/09 dimension reduction PCA/heart disease.csv")  
df.head()

df.describe() 

df.isnull().sum() 

 
def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return (x)

df_norm=norm(df.iloc[:,:])
df_norm.describe() 

z=linkage(df_norm,method="complete",metric="euclidean")

plt.figure(figsize=(15,8)) 
plt.title('Hierarchical Clustering Dendrogram') 
plt.xlabel='index' 
plt.ylabel='Distance'

dendrogram(z,leaf_rotation=0,leaf_font_size=10)


h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 

h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_) 

df['cluster'] = cluster_labels

df=df.iloc[:,:]
  
clus1 = df.iloc[:,:].groupby(df['cluster']).mean() 

##################### KMeans  ################################################

from sklearn.cluster import KMeans  


df1=pd.read_csv("D:/assignments of data science/09 dimension reduction PCA/heart disease.csv")  

 def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return (x)

df_norm1=norm(df1.iloc[:,:])
df_norm1.describe() 


TWSS = []
k = list(range(2, 9))   

for i in k:
    kmeans1 = KMeans(n_clusters = i)  
    kmeans1.fit(df_norm1)    
    TWSS.append(kmeans1.inertia_) 
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS")


model1 = KMeans(n_clusters = 3) 
model1.fit(df_norm1)

model1.labels_ 
mb1 = pd.Series(model1.labels_)  
df1['clust'] = mb1 

df1.head()
df_norm1.head()

df1 = df1.iloc[:,:]
df1.head()

#Grouping by mean of each cluster:
clust2 = df1.iloc[:, :].groupby(df1.clust).mean()

#################################  PCA  #####################################

from sklearn.decomposition import PCA    
from sklearn.preprocessing import scale  



df2=pd.read_csv("D:/assignments of data science/09 dimension reduction PCA/heart disease.csv")   


df2_normal = scale(df2) 
df2_normal

pca = PCA(n_components = 3)     
pca_values = pca.fit_transform(df2_normal)  

# PCA weights
pca.components_  
pca.components_[0]  #pcacomponents for index 0
pca.components_[1]  #pcacomponents for index 1

var = pca.explained_variance_ratio_
var

var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

plt.plot(var1, color = "red")

pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2"


ax = pca_data.plot(x='comp0', y='comp1', kind='scatter',figsize=(12,8))


######################### AgglomerativeClustering ###########################
 
def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return (x)

df_norm2=norm(pca_data.iloc[:,:])
df_norm2.describe() 

#drawing the dendrogram
z=linkage(df_norm2 ,method="complete",metric="euclidean")

plt.figure(figsize=(15,8)) 
plt.title('Hierarchical Clustering Dendrogram') 
plt.xlabel='index' 
plt.ylabel='Distance'

dendrogram(z,leaf_rotation=0,leaf_font_size=10)


h_complete3 = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm2) 

h_complete3.labels_

cluster_labels3 = pd.Series(h_complete3.labels_) 

pca_data['cluster'] = cluster_labels3

pca_data=pca_data.iloc[:,:]
  
clus3= pca_data.iloc[:,:].groupby(pca_data['cluster']).mean() 


############################# KMeans  #######################################

pca_data1 = pd.DataFrame(pca_values)
 
def norm(i):
    x=(i-i.min())/(i.max()-i.min())
    return (x)

df_norm4=norm(pca_data1.iloc[:,:])
df_norm4.describe() 

TWSS1 = []
k = list(range(2, 9)) 

for i in k:
    kmeans2 = KMeans(n_clusters = i) 
    kmeans2.fit(df_norm4)    
    TWSS1.append(kmeans2.inertia_)  
    
TWSS1
# Scree plot 
plt.plot(k, TWSS1, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS")


model2 = KMeans(n_clusters = 3) 
model2.fit(df_norm4)

model2.labels_ 
mb2 = pd.Series(model2.labels_) 
pca_data1['clust'] = mb2 

pca_data1.head()
df_norm4.head()

pca_data1 = pca_data1.iloc[:,:]
pca_data1.head()

clus4 = pca_data1.iloc[:, :].groupby(pca_data1.clust).mean()














































