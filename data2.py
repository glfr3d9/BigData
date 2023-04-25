import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
import numpy as np

#load of dataset on dataframe
df = pd.read_csv('glass.csv')
var_names = list(df.columns) #получение имен признаков
labels = df.to_numpy('int')[:,-1] #метки классов
data = df.to_numpy('float')[:,:-1] #описательные признаки

#normalization of data to the interval from 0 to 1
data = preprocessing.minmax_scale(data)

#construction of a scatterplot of feature pairs
# fig, axs = plt.subplots(2,4)
# for i in range(data.shape[1]-1):
#     axs[i // 4, i % 4].scatter(data[:,i],data[:,(i+1)],c=labels,cmap='hsv')
#     axs[i // 4, i % 4].set_xlabel(var_names[i])
#     axs[i // 4, i % 4].set_ylabel(var_names[i+1])
# plt.show()

#method of principal componentc(PCA)

pca = PCA(n_components = 2)
pca_data = pca.fit(data).transform(data)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv')
plt.show()

#number of components under which variance more than 85%

# pca = PCA(n_components = 5)
# pca_data = pca.fit(data).transform(data)
# print(pca.explained_variance_ratio_)

#inverse_transform method
# pca = PCA(n_components = 2)
# pca_data = pca.fit(data).transform(data)
# inverse_data = pca.inverse_transform(pca_data)
# plt.scatter(inverse_data[:,0],inverse_data[:,1],c=labels,cmap='hsv')
# plt.show()

#varios parameters of svd_solver

# pca = PCA(n_components = 2, svd_solver='randomized')#full, arpack
# pca_data = pca.fit(data).transform(data)
# plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,cmap='hsv')
# plt.show()

#KernelPCA for varios parametrs for kernel

# kernelPCA = KernelPCA(n_components=2,kernel='cosine')#kernel =‘linear’(default), ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’
# kernelPCA_data = kernelPCA.fit(data).transform(data)
# plt.scatter(kernelPCA_data[:,0],kernelPCA_data[:,1],c=labels,cmap='hsv')
# plt.show()

# sparsePCA=SparsePCA(n_components=2)
# sparsePCA_data = sparsePCA.fit(data).transform(data)
# plt.scatter(sparsePCA_data[:,0],sparsePCA_data[:,1],c=labels,cmap='hsv')
# plt.show()

# factorAnalysis = FactorAnalysis(n_components=2)
# factorAnalysis_data = factorAnalysis.fit(data).transform(data)
# plt.scatter(factorAnalysis_data[:,0],factorAnalysis_data[:,1],c=labels,cmap='hsv')
# plt.show()
