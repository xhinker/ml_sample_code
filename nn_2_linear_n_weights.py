# create a nn using numpy to predict linear weights 
import numpy as np
X = np.array(
    [[1,2,3,1]
    ,[2,3,4,1]
    ,[3,4,5,1]
    ,[4,3,2,1]
    ,[12,13,14,1]
    ,[7,8,9,1]
    ,[5,2,10,1]]
    ,dtype=np.float32
)
# a = 2, b = 3, c = 4, d = 5
W_r = np.array([1,20,3,4],dtype=np.float32)
W = np.array([0,0,0,0],dtype=np.float32)
Y = np.array(
    [x@W_r for x in X]
    ,dtype=np.float32
)
print("Y_real\n",Y)

def forward(X):
    return np.dot(X,W)

def loss(Y,Y_pred):
    return ((Y_pred-Y)**2)

def gradient(X,Y,Y_pred):
    return X*(Y_pred-Y)

learning_rate,num_epoch = 0.001,3000

for epoch in range(num_epoch):
    for i,x in enumerate(X):
        y_pred  = forward(x)
        l       = loss(y_pred,Y[i])
        dw      = gradient(x,Y[i],y_pred)
        W       = (W - dw*learning_rate)
    print(f'epoch:{epoch} | loss:{l:.8f}')
print(W)
