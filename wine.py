Name:- Hari singh r
Batch id:-DSWDMCOD 25082022 B

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.pylab as plb

wine=pd.read_csv("D:/assignments of data science/09 dimension reduction PCA/wine.csv")
wine.info()
wine.head()
wine.duplicated()
wine.describe()
wine.isnull().sum()
wine.dtypes

for i in wine.columns:
    plt.hist(wine[i])
    plt.xlabel(i)
    plt.show()

plt.scatter(wine['Alcohol'],wine['Proline'])
plt.xlabel('Alcohol')
plt.ylabel('Proline')


wine_normal=scale(wine)
wine_normal

pca=PCA(n_components=6)
pca_values=pca.fit_transform(wine_normal)

var=pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]

var1=np.cumsum(np.round(var,decimals=4)*100)
var1    


plt.plot(var1,color="red")

pca_values

pca_data=pd.DataFrame(pca_values)
pca_data.columns="comp0","comp1","comp2","comp3","comp4","comp5"

final=pd.concat([wine.Type, pca_data.iloc[:,0:3]],axis=1)
final   

ax=final.plot(x='comp0',y='comp1',kind='scatter',figsize=(12,8),s=90)

final[['comp0', 'comp1','Type']].apply(lambda x: ax.text(*x), axis=1)
    

from scipy.cluster.hierarchy import linkage,dendrogram  

z=linkage(wine_normal,method="ward",metric='euclidean')    
plt.figure(figsize=(15,8))
plt.title("Dendrogram")
plt.xlabel("INDEX")
plt.ylabel("DISTANCE")
dendrogram(z,
           leaf_rotation=0,
           leaf_font_size=10     
           )
plt.show()

from sklearn.cluster import AgglomerativeClustering

h_complete=AgglomerativeClustering(n_clusters=3,linkage="ward",affinity="euclidean").fit(wine_normal)
h_complete.labels_
h_complete.n_clusters   
cluster_labels=pd.Series(h_complete.labels_)
wine['clust']=cluster_labels
wine.head()

wine.iloc[:,1:].groupby(wine.clust).mean()

z1=linkage(final,method="ward",metric="euclidean")
plt.figure(figsize=(15,8))
plt.title("Dendrogram after pca")
plt.xlabel("INDEX")
plt.ylabel("DISTANCE")
dendrogram(z,
           leaf_rotation=0,
           leaf_font_size=10     
           )
plt.show()

h_complete1=AgglomerativeClustering(n_clusters=3,linkage="ward",affinity="euclidean").fit(final)
h_complete1.labels_
h_complete1.n_clusters
cluster_labels1=pd.Series(h_complete1.labels_)
final['clust']=cluster_labels1
x=h_complete.children_
y=h_complete1.children_
final.head()

final.iloc[:,1:].groupby(final.clust).mean()

from sklearn.cluster import KMeans
TWSS=[]
k=list(range(2,9))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(wine_normal)
    TWSS.append(kmeans.inertia_)
    
TWSS
      
plb.plot(k,TWSS,'ro-');plb.xlabel("NO of cluster");plb.ylabel("total with in ss") 

TWSS1= []

k=list(range(2,9))

for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(final)
    TWSS1.append(kmeans.inertia_)
    
TWSS1

plb.plot(k,TWSS1,'ro-');plb.xlabel("NO of cluster");plb.ylabel("Total with in ss")



































