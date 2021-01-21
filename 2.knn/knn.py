import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn")

dfx = pd.read_csv("E:/python/knn Practice/xdata.csv")
dfy = pd.read_csv("E:/python/knn Practice/ydata.csv")

# converting both dfx, dfy data frames into numpy array
X = dfx.values
Y = dfy.values

# frst column in X and Y are serial numbers
X = X[:,1:]
# converting Y into a vector
Y = Y[:,1:].reshape(-1,)


plt.scatter(X[:,0], X[:,1],c = Y)
plt.show()

# distance function that calculates distance between two points
def distance(x1, x2):
    return np.sqrt(sum((x1-x2)**2))

# prediction function A.K.A actual KNN algorithm:
def KNN(x,y,queryPoint, k = 5):
    vals = []
    # m = number of rows in look up data
    m = x.shape[0]

    for i in range(m):
        d = distance(queryPoint, x[i])
        vals.append((d,y[i]))

    # vals now has dist of all points from queryPoint and their outputs
    vals = sorted(vals)
    # now choose k nearest points
    vals = vals[ : k]
    # converting vals into a numpy array
    vals = np.array(vals)
    # get the mapping of all unique outputs and their count
    v = np.unique(vals[ : ,1],return_counts=True)
    # index = index of v that has maximum count
    index = v[1].argmax()
    pred = v[0][index]
    return pred

# sample prdeiction:
q = np.array([0,0])
predY = KNN(X,Y,q)
print(predY)
