import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1 : Data preperation
df = pd.read_csv("E:/python/3.DigitRecognition/train.csv")

# print(df.shape)
# shape of the df is (42000, 785)
# meaning 42000 numbers and each image has 784 (28 * 28) columns representing pixels

# converting df into np.array
data = df.values

# Let X be the pixel data of all numbers in DF
X = data[:,1:]
# Let Y be actual number of pixel datas
Y = data[:,0]

# lets split the 80% of data as training data and 20% as test data
split = (int)(0.8*X.shape[0])
# split = 33600
Xtrain = X[ : split, : ]
Ytrain = Y[ : split]
Xtest = X[split : , : ]
Ytest = Y[split : ]

# visualize data
def drawImage(sample):
    img = sample.reshape(28,28)
    plt.imshow(img, cmap = 'gray')
    plt.show()

# drawImage(Xtrain[3])
# print(Ytrain[3])

# Step 2 : KNN

def distance(x1, x2):
    return np.sqrt(sum((x1-x2)**2))

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

# Step 3 : Predictions
pred = KNN(Xtrain, Ytrain, Xtest[0])
print(int(pred))

# drawImage(Xtest[0])
# print(Ytest[0])

def accuracy():
    # cnt = accuracy got for testing cnt samples
    cnt = 100
    correct = 0
    wrong = 0
    for i in range(cnt):
        pred = int(KNN(Xtrain, Ytrain, Xtest[i]))
        ans = int(Ytest[i])
        if(pred == ans):
            correct += 1
        else:
            wrong += 1
        
    acc = (correct/cnt)*100

    return acc

acc = accuracy()
print(acc)




