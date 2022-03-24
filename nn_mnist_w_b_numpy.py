#%%
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

batch_size  = 100

train_dataset   = torchvision.datasets.MNIST(
    root        = './data'
    ,train      = True
    ,transform  = transforms.ToTensor()
    ,download   = True
)
test_dataset    = torchvision.datasets.MNIST(
    root        = './data'
    ,train      = False
    ,transform  = transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    dataset     = train_dataset
    ,batch_size = batch_size
    ,shuffle    = True
)
test_loader = torch.utils.data.DataLoader(
    dataset     = test_dataset
    ,batch_size = batch_size
    ,shuffle    = False
)

print("load data done")

# define parameters
# w1 is the weight matrix for hidden layer 1, 16 x 784
# 16 layer1 hidden neurals, 784 numbers
# the last column is the bias

learning_rate   = 0.001
num_epoch       = 10
input_size      = 28*28 # 784
layer1_size     = 128
layer2_size     = 64
output_size     = 10

# w1 = np.random.uniform(low = 0,high=1,size=(layer1_size,input_size))#.astype(np.float32)*np.sqrt(1. / layer1_size)
# w2 = np.random.uniform(low = 0,high=1,size=(layer2_size,layer1_size))#.astype(np.float32)*np.sqrt(1. / layer2_size)
# w3 = np.random.uniform(low = 0,high=1,size=(output_size,layer2_size))#.astype(np.float32)*np.sqrt(1. / output_size)
w1 = np.random.randn(layer1_size,input_size) * np.sqrt(1./layer1_size)
b1 = np.random.randn(layer1_size,1) * np.sqrt(1./layer1_size)
w2 = np.random.randn(layer2_size,layer1_size) * np.sqrt(1./layer2_size)
b2 = np.random.randn(layer2_size,1) * np.sqrt(1./layer2_size)
w3 = np.random.randn(output_size,layer2_size) * np.sqrt(1./output_size)
b3 = np.random.randn(output_size,1) * np.sqrt(1./output_size)

def acc():
    total = 0
    correct = 0
    for i,(images,labels) in enumerate(test_loader):
        images = images.numpy()
        labels = labels.numpy()
        for i,image in enumerate(images):
            total +=1
            # prepare raw image data
            image           = image[0].flatten() # channel 0
            image           = np.reshape(image,(input_size,-1))

            # forward to hidden layer 1
            y_pred_layer1_z  = forward(w1,image,b1)
            y_pred_layer1_a  = sigmoid(y_pred_layer1_z)

            # forward to hidden layer 2
            y_pred_layer2_z  = forward(w2,y_pred_layer1_a,b2)
            y_pred_layer2_a  = sigmoid(y_pred_layer2_z)
            #y_pred_layer2   = y_pred_layer2*down_rate

            # output
            y_pred_z        = forward(w3,y_pred_layer2_a,b3)
            y_pred          = softmax(y_pred_z).flatten()
            y_pred          = np.argmax(y_pred) 

            y               = labels[i]
            #print(f"y_pred:{y_pred} | y: {y}")   
            if y_pred == y : correct +=1
    return correct / total

def forward(w,x,b):
    return np.dot(w,x) + b

def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x))/((np.exp(-x)+1)**2)
    return 1/(1 + np.exp(-x))

def softmax(x, derivative=False):
    # Numerically stable with large exponentials
    exps = np.exp(x - x.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)

def ReLU(x):
    return x * (x > 0)

def gradient(x,y_pred,y):
    """x, y_pred and y should be all in 2-D array"""
    return np.dot((y_pred-y),x)


def cross_entropy_loss(p,q):
    return -np.sum(p * np.log(q))

for epoch in range(num_epoch):
    for i,(images,labels) in enumerate(train_loader):
        images = images.numpy()
        labels = labels.numpy()
        for i,image in enumerate(images):
            # prepare raw image data
            image           = image[0].flatten() # channel 0
            image           = np.reshape(image,(input_size,-1))

            # forward to hidden layer 1
            y_pred_layer1_z  = forward(w1,image,b1)
            y_pred_layer1_a  = sigmoid(y_pred_layer1_z)

            # forward to hidden layer 2
            y_pred_layer2_z  = forward(w2,y_pred_layer1_a,b2)
            y_pred_layer2_a  = sigmoid(y_pred_layer2_z)
            #y_pred_layer2   = y_pred_layer2*down_rate

            # output
            y_pred_z        = forward(w3,y_pred_layer2_a,b3)
            y_pred          = softmax(y_pred_z).flatten()
            y               = np.zeros(10)
            y[labels[i]]    = 1
            loss_array      = (y_pred - y) 
            loss            = cross_entropy_loss(y,y_pred)
            #print(f"y_pred:\n{y_pred} \n y:\n {y} \n loss:{loss}" )

            #---------------------------------------

            # update w_l
            loss_array_v    = np.reshape(loss_array,(output_size,-1))
            error           = 2 * loss_array_v*softmax(y_pred_z,derivative=True)
            y_pred_layer2_h = np.reshape(y_pred_layer2_a,(-1,layer2_size))
            dw3             = np.dot(error,y_pred_layer2_h)
            db3             = error
            
            # print("w3",w3.shape)
            # print("w3 t shape",w3.T.shape)

            # update w2  
            error           = np.dot(w3.T,dw3)
            y_l2_d          = sigmoid(y_pred_layer2_z,derivative=True)
            error           = np.dot(error,y_l2_d)
            y_pred_layer1_h = np.reshape(y_pred_layer1_a,(-1,layer1_size))
            dw2             = np.dot(error,y_pred_layer1_h)
            db2             = error
            
            #print(w2.shape)

            # update w1, need d_layer_1
            error           = np.dot(w2.T,dw2) @ sigmoid(y_pred_layer1_z,derivative=True)
            image_h         = np.reshape(image,(-1,input_size))
            dw1             = np.dot(error,image_h)
            db1             = error
            
            w3              = w3 - dw3*learning_rate
            b3              = b3 - db3*learning_rate
            w2              = w2 - dw2*learning_rate
            #b2              = b2 - db2*learning_rate
            w1              = w1 - dw1*learning_rate
            #b1              = b1 - db1*learning_rate
        #print(f"epoch: {epoch} | loss:{loss}")
        #     break
        # break
    print(f'epoch: {epoch} | accuracy: {acc()}')
