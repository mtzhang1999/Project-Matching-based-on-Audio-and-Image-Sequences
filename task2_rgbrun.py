import pickle
import numpy as np
import scipy.signal as signal
import librosa
import scipy
#import noisereduce
import librosa.display
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from PIL import Image
import imageio
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv1_1 = nn.Conv2d(3, 64, 3)
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
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
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

def grade_mode(list):
    list_set=set(list)#取list的集合，去除重复元素
    frequency_dict={}
    for i in list_set:#遍历每一个list的元素，得到该元素何其对应的个数.count(i)
        frequency_dict[i]=list.count(i)#创建dict; new_dict[key]=value
    grade_mode=[]
    for key,value in frequency_dict.items():#遍历dict的key and value。key:value
        if value==max(frequency_dict.values()):
            grade_mode.append(key)
    return grade_mode

def rgb_test(rootpaths):
    cnt=0
    res=[]
    device=torch.device('cuda:0')
    net = Net()
    net.load_state_dict(torch.load('./MTCNN_task2rgb_12.pth'))
    for rootpath in rootpaths:
        gen=[]
        cnt=0
        for parent, dirnames, filenames in os.walk(rootpath +  '/rgb'):
            for filename in filenames:

                imgpath = os.path.join(parent, filename)

                image = default_loader(imgpath)
                transformx = transforms.Compose([transforms.CenterCrop(480),transforms.Resize(188),transforms.ToTensor()])
                image=transformx(image)
                # img = cv2.imread(os.path.join(parent, filename))
                # img = cv2.resize(img, (152,152))
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                # image = torch.from_numpy(img)
                image = image.unsqueeze(0)
                
                with torch.no_grad():
                    outputs = net(image)
                    _, predicted = torch.max(outputs.data, 1)
                    gen.append(int(predicted[0]))
                    cnt=cnt+1
                if cnt==8:
                    break
            break
        print(gen,grade_mode(gen))
        res.append(grade_mode(gen))
    return res