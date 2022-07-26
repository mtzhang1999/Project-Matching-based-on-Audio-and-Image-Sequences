import torch.nn.functional as F
import torch
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt
import pickle
import numpy as np
import json

root =os.getcwd()
def default_loader(path):
        return Image.open(root+path)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.batch1=nn.BatchNorm2d(64,affine=True)
        self.batch2=nn.BatchNorm2d(128,affine=True)
        self.batch3=nn.BatchNorm2d(256,affine=True)
        self.batch4=nn.BatchNorm2d(512,affine=True)
        self.conv1_1 = nn.Conv2d(4, 64, 3)
        self.conv1_2 = nn.Conv2d(64, 64, 3)
        self.conv2_1 = nn.Conv2d(64,128,3)
        self.conv2_2 = nn.Conv2d(128,128,3)
        self.conv3_1 = nn.Conv2d(128,256,3)
        #self.conv3_2 = nn.Conv2d(256,256,3)
        self.conv3_3 = nn.Conv2d(256,256,3)
        self.conv4_1 = nn.Conv2d(256,512,3)
        self.conv4_2 = nn.Conv2d(512,512,3)
        #self.conv4_3 = nn.Conv2d(512,512,3)
        self.conv4_4 = nn.Conv2d(512,512,3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.batch1(F.relu(self.conv1_1(x)))
        x = self.pool1(F.relu(self.conv1_2(x)))
        x = self.batch2(F.relu(self.conv2_1(x)))
        x = self.pool2(F.relu(self.conv2_2(x)))
        x = self.batch3(F.relu(self.conv3_1(x)))
        #x = F.relu(self.conv3_2(x))
        x = self.pool3(F.relu(self.conv3_3(x)))
        x = self.batch4(F.relu(self.conv4_1(x)))
        x = F.relu(self.conv4_2(x))
        #x = F.relu(self.conv4_3(x))
        x = self.pool4(F.relu(self.conv4_4(x)))
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def task2(imgs):
    transformx = transforms.Compose([transforms.Resize(188),transforms.ToTensor()])
    res=[]
    net = Net()
    net.load_state_dict(torch.load('./MTCNN_task2.pth'))
    for img in imgs:

        img0 = default_loader(img[0])
        img1 = default_loader(img[1]) 
        img2 = default_loader(img[2]) 
        img3 = default_loader(img[3])
        img0 = transformx(img0)
        img1 = transformx(img1) 
        img2 = transformx(img2) 
        img3 = transformx(img3) 
        img4=torch.ones(4,188,188)
        img4[0,:,:]=img0
        img4[1,:,:]=img1
        img4[2,:,:]=img2
        img4[3,:,:]=img3
        
        with torch.no_grad():
            inputs = img4.unsqueeze(0)
            outputs = net(inputs)[0]
        outputs[0]*=20
        outputs[1]*=20
        res.append(outputs.tolist())
        print(outputs.tolist())
    return res