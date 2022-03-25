import numpy as np
# model = y = 2*x + 3
X = np.array([1,2,3,4],dtype=np.float32)
Y = [(2*x + 3) for x in X]
w,b = 0.0,0.0

def forward(x):
    return w*x+b

def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

def gradient(x,y,y_pred):
    return np.dot(x,(y_pred-y))

print(f'print the prediction before training: f(5)={forward(5)}')

learning_rate,n_inters = (0.01,10000)

for epoch in range(n_inters):
    y_pred  = forward(X) 
    l       = loss(Y,y_pred)
    dw      = gradient(X,Y,y_pred)
    w       = w - learning_rate*dw
    b       = b + learning_rate*l
    print(f'epoch {epoch+1}: w= {w:.3f},b= {b:.3f} loss = {l:.8f}, dw = {dw:.3f}')

print(f'print the prediction after training: f(5)={forward(5)}')
