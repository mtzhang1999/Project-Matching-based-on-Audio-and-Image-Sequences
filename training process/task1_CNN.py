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

root =os.getcwd()
def default_loader(path):
        return Image.open(root+path)
    
class MyDataset(Dataset): 
    def __init__(self,transform=None,target_transform=None, loader=default_loader):
	                          
        super(MyDataset,self).__init__()
        imgs=[]
        self.classes={'061_foam_brick': 0,
        'green_basketball': 1,
        'salt_cylinder': 2,
        'shiny_toy_gun': 3,
        'stanley_screwdriver': 4,
        'strawberry': 5,
        'toothpaste_box': 6,
        'toy_elephant': 7,
        'whiteboard_spray': 8,
        'yellow_block': 9}
        # for key,value in self.classes.items():
        #     f=os.walk(root+'/train/'+key+'/'+key)
        #     for i,j,k in f:
        #         for num in j:
        #             img_t=[]
        #             for n in range(4):
        #                 img_t.append('/train/'+key+'/'+key+'/'+num+'/ceptrum'+str(n)+'.jpg')
        #             imgs.append((img_t,value))
        #         break
        # m=np.array(imgs)
        # np.save('./data.npy',m)
        self.imgs = np.load('data.npy',allow_pickle=True).tolist()
        # self.imgs=imgs
        #print(len(self.imgs))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img4=[]
        img0 = self.loader(fn[0])
        img1 = self.loader(fn[1]) 
        img2 = self.loader(fn[2]) 
        img3 = self.loader(fn[3])
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1) 
            img2 = self.transform(img2) 
            img3 = self.transform(img3) 
        img4=torch.ones(4,188,188)
        img4[0,:,:]=img0
        img4[1,:,:]=img1
        img4[2,:,:]=img2
        img4[3,:,:]=img3
        #print(img4.shape)
        return img4,label
    def __len__(self):
        return len(self.imgs)

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
        self.fc3 = nn.Linear(128, 10)

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

if __name__ == "__main__":
    device=torch.device('cuda:0')
    transformx = transforms.Compose([transforms.Resize(188),transforms.ToTensor()])
    trainset = MyDataset(transform=transformx)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
    classes=['061_foam_brick',
        'green_basketball',
        'salt_cylinder',
        'shiny_toy_gun',
        'stanley_screwdriver',
        'strawberry',
        'toothpaste_box',
        'toy_elephant',
        'whiteboard_spray',
        'yellow_block']
    net = Net()
    net.to(device)
    x=[]
    loss_array=[]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003, betas=(0.9,0.99))
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
    for epoch in range(40):

        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            #print(inputs.shape)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss+=loss.item()/200
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0
        loss_array.append(epoch_loss)
        torch.save(net.state_dict(),  './MTCNN_task1g_'+str(epoch)+'.pth')
        x.append(epoch+1)
    print('Finished Training')
    PATH = './MTCNN_task1g.pth'
    torch.save(net.state_dict(), PATH)
    plt.plot(x,loss_array,label='training loss',color='r',linewidth=1)
    plt.legend
    plt.savefig('./loss.png')