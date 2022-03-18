# Use plain Python code using Numpy to build a workable NN model
import numpy as np
X       = np.array([1,2,3,4],dtype=np.float32)
w,w_r   = 0.0,2.0
Y       = np.array([(x*w_r) for x in X],dtype=np.float32)
learning_rate,num_epoch = 0.01,100
for epoch in range(num_epoch):
    for i,x in enumerate(X):
        y_pred  = w*x                   # aka forward move, to predict value based on a random weight
        dw      = (y_pred - Y[i])*x     # the gradient dw which will be used to update the current w
        w       = w - learning_rate*dw  # update the w
    print(f'epoch:{epoch+1} | w = {w:.3f}')
print("predicted w:",w)
