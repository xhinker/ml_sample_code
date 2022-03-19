import enum
from re import I
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 

# prepare data
bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target
n_samples,n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234)

print(y_train)

# transform to the standard dataset
sc      = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
y_train = np.where(y_train==0,-1,y_train)
y_test  = np.where(y_test==0,-1,y_test)
print(y_train,y_test)

# define the target 
W = np.ones(30)
learning_rate,num_epoch = 0.001,10000

def forward(x):
    return np.dot(x,W)

def loss(Y,Y_pred):
    return ((Y_pred-Y)**2)

def gradient(X,Y,Y_pred):
    return X*(Y_pred-Y)

# train the data
for epoch in range(num_epoch):
    for i,x in enumerate(X_train):
        y_pred  = forward(x)
        l       = loss(y_pred,y_train[i])
        dw      = gradient(x,y_train[i],y_pred)
        W       = (W - dw*learning_rate)
    #print(f'epoch:{epoch} | loss:{l:.8f}')

print(W)

# test the result
correct = 0
for i,x in enumerate(X_test):
    y_pred_raw  = forward(x)
    y_pred      = 1 if y_pred_raw>=0 else -1

    if y_pred==y_test[i]:
        correct += 1
    else:
        print(f'y_pred_raw:{y_pred_raw}| y_pred:{y_pred} | y:{y_test[i]}')
        continue

print(correct/len(y_test))
