import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


#copy data x1,x2 from clustering_python.ods
x=np.array([[1.2,2.0],[1.5,1.8],[2.4,3.0],[5.8,8.1],[6.3,4.9],[8.6,7.5],[3.1,3.3]])
#plt.scatter(x[:,0],x[:,1], s=50)
#plt.show()

#There are two types of hierarchical clustering:
#agglomerative or ‘bottom up’ and divisive
#or ‘top down’.
#With divisive clustering we start from a situation
#where all observations are in the same cluster.
#agglomerative clustering has an approach bottom up.
#we start with each case being its own cluster.
#There is a total of N clusters.
#Second, using some similarity measure like
#Euclidean distance, we group the two closest
#clusters together, reaching an ‘n minus
#1’ cluster solution.
#Then we repeat this procedure, until all observations
#are in a single cluster.The name for this type of graph is: a ‘dendrogram’.
linkage_matrix=linkage(x, "single")
dendrogram=dendrogram(linkage_matrix,truncate_mode="none")
plt.title("hierarchical clustering agglomerative ")
plt.show()
