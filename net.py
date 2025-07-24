import torch.nn.functional as F
import torch

from torch import nn

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.conv1 = nn.Conv2d(3, 96, 11, 4)  
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2) 
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)   
        self.fc2 = nn.Linear(4096, 4096)   
        self.fc3 = nn.Linear(4096, 1000)  

    def forward(self, x):
        x = F.relu(self.conv1(x))         
        x = F.max_pool2d(x, 3, 2)             
        x = F.relu(self.conv2(x))        
        x = F.max_pool2d(x, 3, 2)  
        x = F.relu(self.conv3(x)) 
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x))  
        x = F.max_pool2d(x, 3, 2)                
        x = torch.flatten(x, 1)         
        x = F.relu(self.fc1(x))   
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))   
        x = F.dropout(x, 0.5)      
        x = self.fc3(x)          

        return x
