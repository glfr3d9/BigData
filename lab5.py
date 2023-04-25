import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances_argmin

print('1,2. Создать Python скрипт. Загрузить данные в датафрейм')
data = pd.read_csv('iris.data', header=None)
print(data)
# в методичке не было куска кода с преобразованием данных, пришлось копать. Работает и чёрт с ним
no_labeled_data = x = data.iloc[:, [0, 1, 2, 3]].values
print(no_labeled_data)
print('KMEANS')
print('1. Проведем кластеризацию методов k-средних')
k_means = KMeans(init='k-means++', n_clusters=3, n_init=15)
k_means.fit(no_labeled_data)
print(k_means)
print('2. Получим центры кластеров и определим какие наблюдения в какой кластер попали')
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(no_labeled_data, k_means_cluster_centers)
print(k_means_cluster_centers)
print(k_means_labels)
print('3. Построим результаты классификации для признаков попарно (1 и 2, 2 и 3, 3 и 4)')
f, ax = plt.subplots(1, 3)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
print(ax)
for i in range(3):
    my_members = k_means_labels == i
    cluster_center = k_means_cluster_centers[i]
    for j in range(3):
        ax[j].plot(no_labeled_data[my_members, j], no_labeled_data[my_members, j + 1], 'w', markerfacecolor=colors[i],
                   marker='o', markersize=4)
        ax[j].plot(cluster_center[j], cluster_center[j + 1], 'o', markerfacecolor=colors[i], markeredgecolor='k',
                   markersize=8)
plt.show()
print('Уменьшите размерность данных до 2 используя метод главных компонент и нарисуйте карту для всей области значений')
reduced_data = PCA(n_components=2).fit_transform(no_labeled_data)
kmeans = KMeans(init="k-means++", n_clusters=3, n_init=15)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

print('5. Исследуйте работу алгоритма k-средних при различных параметрах init')
k_means = KMeans(init='k-means++', n_clusters=3, n_init='auto')
k_means.fit(no_labeled_data)
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(no_labeled_data, k_means_cluster_centers)
print(k_means_cluster_centers)
print(k_means_labels)
f, ax = plt.subplots(1, 3)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
print(ax)
for i in range(3):
    my_members = k_means_labels == i
    cluster_center = k_means_cluster_centers[i]
    for j in range(3):
        ax[j].plot(no_labeled_data[my_members, j], no_labeled_data[my_members, j + 1], 'w', markerfacecolor=colors[i],
                   marker='o', markersize=4)
        ax[j].plot(cluster_center[j], cluster_center[j + 1], 'o', markerfacecolor=colors[i], markeredgecolor='k',
                   markersize=8)
plt.show()
print('6. Определите наилучшее количество методом локтя (консоль выдаёт тучу варнингов но работает)')
wcss = []
for i in range(1, 15):
    kmean = KMeans(n_clusters=i, init="k-means++")
    kmean.fit_predict(no_labeled_data)
    wcss.append(kmean.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()
print('7. Проведите кластеризацию используя пакетную кластеризацию k-средних')
k_means = MiniBatchKMeans(n_clusters=3,
                          random_state=0,
                          batch_size=6,
                          max_iter=10,
                          n_init="auto").fit(no_labeled_data)
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(no_labeled_data, k_means_cluster_centers)
f, ax = plt.subplots(1, 3)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
print(ax)
for i in range(3):
    my_members = k_means_labels == i
    cluster_center = k_means_cluster_centers[i]
    for j in range(3):
        ax[j].plot(no_labeled_data[my_members, j], no_labeled_data[my_members, j + 1], 'w', markerfacecolor=colors[i],
                   marker='o', markersize=4)
        ax[j].plot(cluster_center[j], cluster_center[j + 1], 'o', markerfacecolor=colors[i], markeredgecolor='k',
                   markersize=8)
plt.show()
print('Иерархическая кластеризация')
print('1. Проведем иерархическую кластеризацию на тех же данных')
hier = AgglomerativeClustering(n_clusters=3, linkage='average')
hier = hier.fit(no_labeled_data)
hier_labels = hier.labels_
print('2. Отобразим результаты кластеризации')
f, ax = plt.subplots(1, 3)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
for i in range(3):
    my_members = hier_labels == i
    for j in range(3):
        ax[j].plot(no_labeled_data[my_members, j], no_labeled_data[my_members, j + 1], 'w', markerfacecolor=colors[i],
                   marker='o', markersize=4)
plt.show()
print('5. Сгенерируйте случайные данные в виде двух колец')
data1 = np.zeros([250, 2])
for i in range(250):
    r = random.uniform(1, 3)
    a = random.uniform(0, 2 * math.pi)
    data1[i, 0] = r * math.sin(a)
    data1[i, 1] = r * math.cos(a)
data2 = np.zeros([500, 2])
for i in range(500):
    r = random.uniform(5, 9)
    a = random.uniform(0, 2 * math.pi)
    data2[i, 0] = r * math.sin(a)
    data2[i, 1] = r * math.cos(a)
    data = np.vstack((data1, data2))
print('6. Проведите иерархическую кластеризацию')
hier = AgglomerativeClustering(n_clusters=2, linkage='ward')
hier = hier.fit(data)
hier_labels = hier.labels_
print('7. Выведите полученные результаты')
my_members = hier_labels == 0
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o',
         markersize=4,
         color='red', linestyle='None')
my_members = hier_labels == 1
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o',
         markersize=4,
         color='blue', linestyle='None')
plt.show()
print('Консоль ругается но мы живём')
