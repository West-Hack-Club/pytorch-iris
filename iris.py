from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

class TextLoading:
    
    def __init__(self, filepath):
        self.filepath = filepath

    def load(self):
        loadedText = np.genfromtxt(self.filepath, delimiter=',',skip_header=1)
        return loadedText

X = TextLoading('C:/PythonProjects/irispytorch/data/getdata.csv')
X = X.load()

y = TextLoading('C:/PythonProjects/irispytorch/data/datacls.csv')
y = y.load()

#X, y = load_iris(return_X_y=True)

clf = LogisticRegression(random_state=0, max_iter=9999).fit(X, y)
pred = clf.predict(X)

clf.predict_proba(X[:2, :])
clf.score(X, y)