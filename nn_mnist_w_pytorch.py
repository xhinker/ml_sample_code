#%% 
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# hyper parameters 
input_size      = 784 # 28x28
hidden_size     = 50 
num_classes     = 10 # output size
num_epochs      = 6
batch_size      = 300
learning_rate   = 0.001

# prepare MNIST data
train_dataset = torchvision.datasets.MNIST(root='./data'
                                            ,train=True
                                            ,transform=transforms.ToTensor()
                                            ,download=True)
test_dataset = torchvision.datasets.MNIST(root='./data'
                                          ,train=False
                                          ,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset
                                           ,batch_size=batch_size
                                           ,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset
                                          ,batch_size=batch_size
                                          ,shuffle=False)

# grab a sample to see what the data looks like
examples = iter(train_loader)
samples, labels = examples.next()
print(samples.shape,labels.shape)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap='gray')
#plt.show()
print(labels)

#%% my NN 
class AZ_NN(nn.Module):
    def __init__(self,epochs=10,sizes=[784,16,16,10]):
        super(AZ_NN,self).__init__()
        self.sizes  = sizes
        self.epochs = epochs
        self.layer1 = nn.Linear(self.sizes[0],self.sizes[1])
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(self.sizes[1],self.sizes[2])
        self.relu   = nn.ReLU()
        self.layer3 = nn.Linear(self.sizes[2],self.sizes[3])

    def forward(self,x):
        x   = self.layer1(x)
        x   = self.relu(x)
        x   = self.layer2(x)
        x   = self.relu(x)
        out = self.layer3(x)
        return out

    def train(self,train_loader,optimizer,loss):
        total_step = len(train_loader)
        for epoch in range(self.epochs):
            for i,(images,labels) in enumerate(train_loader):
                images  = images.reshape(-1,28*28)
                outputs = self.forward(images)
                l       = loss(outputs,labels)              # pytorch will handle one hot encoding
                l.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if (i+1)%100 == 0:
                    print(f'epoch #:{epoch}/{self.epochs} | step #:{i+1}/{total_step} | loss:{l:.4f}')

model       = AZ_NN(epochs=40,sizes=[784,128,64,10])
loss        = nn.CrossEntropyLoss()
optimizer   = torch.optim.Adam(model.parameters(),lr=learning_rate)
model.train(train_loader,optimizer=optimizer,loss=loss)

#%% test model 
# Test the model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images,labels in test_loader:
        images = images.reshape(-1,28*28)
        #labels = labels.to(device)
        outputs = model.forward(images)
        _,predicted = torch.max(outputs.data,1) # need figure this out in detail
        n_samples += labels.size(0)
        n_correct += (predicted==labels).sum().item() # this is amazing operation, saved so many lines of code

acc = 100.0 * n_correct / n_samples
print(f'acc:{acc}')
