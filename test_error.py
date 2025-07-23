import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from cnn_scratch import Conv2D, MaxPool2D, FNN, relu, flatten

torch.manual_seed(0)

class CNN_pytorch(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  
        self.conv2 = nn.Conv2d(32, 64, 3, 1) 
        self.fc1 = nn.Linear(64*5*5, 128)   
        self.fc2 = nn.Linear(128, 10)  

        global w_conv1, w_conv2, b_conv1, b_conv2, w_fc1, w_fc2, b_fc1, b_fc2

        w_conv1 = self.conv1.weight      
        w_conv2 = self.conv2.weight      
        b_conv1 = self.conv1.bias      
        b_conv2 = self.conv2.bias    
        w_fc1 = self.fc1.weight
        w_fc2 = self.fc2.weight
        b_fc1 = self.fc1.bias
        b_fc2 = self.fc2.bias

    def forward(self, x):
        x = F.relu(self.conv1(x))         
        x = F.max_pool2d(x, 2)             
        x = F.relu(self.conv2(x))        
        x = F.max_pool2d(x, 2)                 
        x = torch.flatten(x, 1)         
        x = F.relu(self.fc1(x))          
        x = self.fc2(x)          

        return x

class CNN_scratch():
    def __init__(self):       
        self.conv1 = Conv2D(3, 32, 3, 1)  
        self.conv2 = Conv2D(32, 64, 3, 1) 
        self.max_pool = MaxPool2D(2, 2)
        self.fc1 = FNN(64*5*5, 128)   
        self.fc2 = FNN(128, 10)  

        self.conv1.weights = w_conv1.detach().numpy()
        self.conv2.weights = w_conv2.detach().numpy()
        self.conv1.bias = b_conv1.detach().numpy()
        self.conv2.bias = b_conv2.detach().numpy()  
        self.fc1.w = w_fc1.T.detach().numpy()
        self.fc2.w = w_fc2.T.detach().numpy()
        self.fc1.b = b_fc1.detach().numpy()
        self.fc2.b = b_fc2.detach().numpy()

    def __call__(self, x):
        x = relu(self.conv1(x))             
        x = self.max_pool(x)               
        x = relu(self.conv2(x))        
        x = self.max_pool(x)     
        x = flatten(x)           
        x = relu(self.fc1(x))           
        x = self.fc2(x) 

        return x
    
cnn_pytorch = CNN_pytorch()
cnn_scratch = CNN_scratch()

x = torch.randn(2, 3, 28, 28)

o1 = cnn_pytorch(x)
o2 = cnn_scratch(x)

print(o1.shape, o2.shape)
print(torch.allclose(o1, torch.from_numpy(o2).to(o1.dtype), atol=1e-5))