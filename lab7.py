import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


data = pd.read_csv('iris.data',header=None)

X = data.iloc[:,:4].to_numpy()
labels = data.iloc[:,4].to_numpy()

le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("GaussianNB")
print((y_test != y_pred).sum()) #количество наблюдений, который были неправильно определены
print("Точность классификации: ", gnb.score(X_test, y_test)*100)


# x = []
# score = []
# count = []
# i=0.05
# while i<1:
#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=i,
#                                                         random_state=20046)
#     gnb = GaussianNB()
#     y_pred = gnb.fit(X_train, y_train).predict(X_test)
#     x.append(i)
#     score.append(gnb.score(X_test, y_test)*100)
#     count.append((y_test != y_pred).sum())
#     i+=0.05
# plt.plot(x, score)
# plt.plot(x, count)
# plt.show()

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
y_pred = mnb.fit(X_train, y_train).predict(X_test)
print("MultinomialNB")
print((y_test != y_pred).sum()) #количество наблюдений, который были неправильно определены
print("Точность классификации: ", mnb.score(X_test, y_test)*100)

from sklearn.naive_bayes import ComplementNB
cnb = ComplementNB()
y_pred = cnb.fit(X_train, y_train).predict(X_test)
print("ComplementNB")
print((y_test != y_pred).sum()) #количество наблюдений, который были неправильно определены
print("Точность классификации: ", cnb.score(X_test, y_test)*100)

from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
y_pred = bnb.fit(X_train, y_train).predict(X_test)
print("BernoulliNB")
print((y_test != y_pred).sum()) #количество наблюдений, который были неправильно определены
print("Точность классификации: ", bnb.score(X_test, y_test)*100)

print("Классифицирующие деревья")
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion="gini",
                                  splitter="random",
                                  max_depth=4,
                                  min_samples_split=3,
                                  min_samples_leaf=3)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print((y_test != y_pred).sum())
print("Точность классификации: ", clf.score(X_test, y_test)*100)
print("Количество листьев: ", clf.get_n_leaves())
print("Глубина дерева: ", clf.get_depth())

import matplotlib.pyplot as plt
plt.subplots(1,1,figsize = (10,10))
tree.plot_tree(clf, filled = True)
plt.show()

x = []
score = []
count = []
i=0.05
while i<1:
    clf = tree.DecisionTreeClassifier(criterion="gini",
                                      splitter="random",
                                      max_depth=4,
                                      min_samples_split=3,
                                      min_samples_leaf=3)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    x.append(i)
    score.append(gnb.score(X_test, y_test)*100)
    count.append((y_test != y_pred).sum())
    i+=0.05
plt.plot(x, score)
plt.plot(x, count)
plt.show()
