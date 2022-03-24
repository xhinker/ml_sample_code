from sklearn import datasets
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
import numpy as np

# prepare data
bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target
n_samples,n_features = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

# transform to the standard dataset
sc      = StandardScaler()
X_train = sc.fit_transform(X_train)
#X_train = np.hstack((X_train,np.ones((X_train.shape[0],1)))) 
X_test  = sc.transform(X_test)
#X_test = np.hstack((X_test,np.ones((X_test.shape[0],1))))
y_train = np.where(y_train==0,-1,y_train) # replace 0 with -1
y_test  = np.where(y_test==0,-1,y_test)   # replace 0 with -1

# initialize weights and hyper parameters 
W = np.random.randn(30).astype(np.float32)
learning_rate,num_epoch = 0.0001,10000

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

# test the result
def acc(x,y):
    correct = 0
    for i,x in enumerate(x):
        y_pred_raw  = forward(x)
        y_pred      = 1 if y_pred_raw>=0 else -1

        if y_pred==y[i]:
            correct += 1
        else:
            print(f'y_pred_raw:{y_pred_raw}| y_pred:{y_pred} | y:{y[i]}')
            continue
    return correct/len(y)

print('train acc:',acc(X_train,y_train))
print('test acc:',acc(X_test,y_test))
