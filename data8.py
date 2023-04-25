import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from mlxtend.preprocessing import TransactionEncoder

data = pd.read_csv('iris.data',header=None)
df = pd.DataFrame(data)
print(df)

# Выделение данных и их меток
X = data.iloc[:,:4].to_numpy()
labels = data.iloc[:,4].to_numpy()

#Преобразование текстов меток к числам
le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)

#обучающая и тестовая выборки
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

#Классификация наблюдений LDA
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

clf = LinearDiscriminantAnalysis()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print("Неправильные наблюдения:: "+str((y_test != y_pred).sum())) #количество наблюдений, который были неправильно определены

#точность классификации
print('Точность классификации: '+str(clf.score(X_test, y_test)))

wrong = []
accuracy = []
for i in range(5,95,5):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=10080)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    wrong.append((y_test != y_pred).sum())
    accuracy.append(clf.score(X_test, y_test))
plt.scatter(wrong, accuracy)

df['size']=df[4].transform(len)
print('Transform')
print(df)
print()

# Изменные параметры классификации
clf = LinearDiscriminantAnalysis('lsqr','auto')
y_pred = clf.fit(X_train, y_train).predict(X_test)
print('Измененные параметры классификации')
print("Неправильные наблюдения: "+str((y_test != y_pred).sum()))
print('Точность классификации: '+str(clf.score(X_test, y_test)))

# Классификация SVC
clf = svm.SVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print('SCV')
print('Неправильные измерения '+str((y_test != y_pred).sum()))
print('Точность ' + str(clf.score(X, Y)))

print('Точность классификации: '+str(clf.score(X_test, y_test)))


wrong1 = []
accuracy1 = []
for i in range(5,95,5):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i, random_state=10080)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    wrong.append((y_test != y_pred).sum())
    accuracy.append(clf.score(X_test, y_test))
plt.scatter(wrong1, accuracy1)

print('Работа вектора при различных параметрах')
clf = svm.SVC(kernel='linear', degree=2,max_iter=100)
y_pred = clf.fit(X_train, y_train).predict(X_test)

print('Неправильные измерения '+str((y_test != y_pred).sum()))
print('Точность ' + str(clf.score(X, Y)))

print('Исследование с NuSVC')
clf = svm.NuSVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print('Неправильные измерения '+str((y_test != y_pred).sum()))
print('Точность ' + str(clf.score(X, Y)))

print('Исследование с LinearSVC')
clf = svm.LinearSVC(max_iter=10000)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print('Неправильные измерения '+str((y_test != y_pred).sum()))
print('Точность ' + str(clf.score(X, Y)))


# plt.show()