
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import precision_score as pc
from sklearn.metrics import recall_score as rc


import matplotlib.pyplot as plt

csvpath = 'C:/PythonProjects/irispytorch/data/'

class TextLoading:
    
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        loadedText = np.genfromtxt(self.filepath, delimiter=',',skip_header=1)
        return loadedText

X = TextLoading('C:/PythonProjects/irispytorch/data/getdata.csv')
X = X.load()

y = TextLoading('C:/PythonProjects/irispytorch/data/clsround.csv')
y = y.load()


clf = lr(random_state=0).fit(X, y)

pred = clf.predict(X)

y[:20]

predval = clf.predict_proba(X)

p1 = predval[:, 1]

np.mean(p1)
np.std(p1)
np.percentile(p1,100)
np.max(p1)

p2 = np.where(p1 < 0.25, 0, 1)

plt.hist(p2,bins=100)
plt.show()

class perform:
    def __init__(self,ytrue,predicted) -> None:
        self.predicted = predicted
        self.ytrue = ytrue

    def acc(self):
        acc = ac(self.ytrue,self.predicted)
        return acc

    def precision(self):
        precision = pc(self.ytrue,self.predicted, average = 'micro')
        return precision

    def recall(self):
        recall = rc(self.ytrue,self.predicted, average = 'micro')
        return recall

perf = perform(y,pred)

perf.acc()

perf.precision()

perf.recall()

        
a = 0
b = 0
c = 0
d = 0

for i in range(len(y)):
    if y[i] == 0 and pred[i] == 0:
        print("true negative")
        a = a + 1
    elif y[i] == 0 and pred[i] == 1:
        print("false positive")
        b = b + 1
    elif y[i] == 1 and pred[i] == 0:
        print("false negative")
        c = c + 1
    elif y[i] == 1 and pred[i] == 1:
        print("true positive")
        d = d + 1
    else:
        print("error")

(d)/(d+b)
(d)/(d+c)

pc(y,pred)
rc(y,pred)