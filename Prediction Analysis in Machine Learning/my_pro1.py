#load libraries
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
import pandas as pd
import random
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
import pyodbc

#load dataset
df = pd.read_csv('E:\python\cancer.txt')
conn = pyodbc.connect(r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=E:\python\py_db_1.accdb;')
cursor=conn.cursor()

#data description
print(df.dtypes)
print(df.shape)
print(df.head(20))
print(df.describe())

#graphical representation
df.hist()
plt.show()

#data cleaning
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
print(full_data)

#dataset={2:[[[5.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 1.0],[2,3],[3,1]],4:[[6,5],[7,7],[8,6]]}

#train and test datasets
random.shuffle(full_data)
test_size = 0.2
train_set = {2:[], 4:[]} #2 & 4 is output data
test_set = {2:[], 4:[]}  #2 is for the benign tumors   4 is for malignant tumors,
train_data = full_data[0:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]
for data in train_data:
    train_set[data[-1]].append(data[0:-1])
for data in test_data:
    test_set[data[-1]].append(data[0:-1])
test_set

#defining KNN algorithm from scratch
def knn(data, predict, k=3):
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]

    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result   # 2,4

correct=0
total =0
#new training datasets
for i  in test_set:#4
    for pt in test_set[i]:
        pr = knn(train_set,pt,200) 
        if (pr == i):
            correct +=1
        total +=1
print(correct/total)
correct = 0
total=0
for group in test_set:
    for data in test_set[group]:
        print(group,end='->')
        print(data,end='->')
        vote = knn(train_set, data, k=6)
        print(vote)
        
        if group == vote: 
            correct += 1
        total += 1
print('Accuracy:', correct/total)

#user defined input
new_pt=[]
id=int(input('enter the id :'))
i1=int(input('x1 :'))
i2=int(input('x2 :'))
i3=int(input('x3 :'))
i4=int(input('x4 :'))
i5=int(input('x5 :'))
i6=int(input('x6 :'))
i7=int(input('x7 :'))
i8=int(input('x8 :'))
i9=int(input('x9 :'))
new_pt.append(i1)
new_pt.append(i2)
new_pt.append(i3)
new_pt.append(i4)
new_pt.append(i5)
new_pt.append(i6)
new_pt.append(i7)
new_pt.append(i8)
new_pt.append(i9)
print('the data entered is  :',new_pt)
s=knn(train_set,new_pt,500)
print('values predicted :',s)

#database

cursor.execute('''INSERT INTO py_tb VALUES ((?),(?),(?),(?),(?),(?),(?),(?),(?),(?),(?))''',(id),(i1),(i2),(i3),(i4),(i5),(i6),(i7),(i8),(i9),s)
conn.commit();
                    
