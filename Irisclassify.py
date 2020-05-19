#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#人认识数据
#print("keys of iris_dataset: \n{}".format(iris_dataset.keys()))
#print(iris_dataset['DESCR'][:193]+"\n...")
#print("First five rows of data: \n{}".format(iris_dataset['data'][:5]))

#留出训练数据和测试数据
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0) #random_state指定了随机数生成器的种子
#print("X_train shape: {}".format(X_train.shape))
#print("X_test shape: {}".format(X_test.shape))

#观察数据，做散点矩阵图
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
#plt.show()

#k近邻算法
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
#做出预测
X_new = np.array([[5,2.9,1,0.2]]) #预测花萼长5cm宽2.9cm，花瓣长1cm宽0.2cm
#print("X_new.shape:{}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))
#评估模型。测试集进行预测
y_pred = knn.predict(X_test)
print("Test set predictions:\n{}".format(y_pred))
print("Test set score:{:.2f}".format(np.mean(y_pred==y_test)))#计算精度。精度是预测正确的花所占的比例
#print("Test set score:{:.2f}".format(knn.score(X_test,y_test)))   #还可以用knn的对象score方法来计算精度

