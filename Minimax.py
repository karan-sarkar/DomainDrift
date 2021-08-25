import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

import tqdm
import itertools
import random

BATCH_SIZE = 32

lr = float(input("lr"))
beta1 = 0.5
beta2 = 0.999

#USPS -> MNIST 0.992
#SVHN -> MNIST 0.985

a = int(input("a"))
b = int(input("b"))
conf = float(input("confidence"))

SOURCE_CHANNELS = 1
TARGET_CHANNELS = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trans = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
 ])
 
mnist = torchvision.datasets.MNIST(root = 'mnist/', train = True, download = True, transform = trans)
mnist_test = torchvision.datasets.MNIST(root = 'mnist/', train = False, download = True, transform = trans)
usps = torchvision.datasets.USPS(root = 'usps/', train = True, download = True, transform = trans)
usps_test = torchvision.datasets.USPS(root = 'usps/', train = False, download = True, transform = trans)
svhn = torchvision.datasets.SVHN(root = 'svhn/', split = 'train', download = True, transform = trans)
svhn_test = torchvision.datasets.SVHN(root = 'svhn/', split = 'test', download = True, transform = trans)

def get_source(train, cycle):
    if train:
        data = enumerate(torch.utils.data.DataLoader(svhn, batch_size=BATCH_SIZE, shuffle=True))
    else:
        data = enumerate(torch.utils.data.DataLoader(svhn_test, batch_size=BATCH_SIZE, shuffle=True))
    if cycle:
        data = itertools.cycle(data)
    return data

def get_target(train, cycle):
    if train:
        data = enumerate(torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True))
    else:
        data = enumerate(torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True))
    if cycle:
        data = itertools.cycle(data)
    return data

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class G(nn.Module):
    def __init__(self, conv_dim=64, use_labels=False):
        super(G, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
       
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 8192)
        return x
        
class C(nn.Module):
    def __init__(self):
        super(C, self).__init__()
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.fc1 = nn.Linear(10, 15)
        self.fc2 = nn.Linear(15, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
       
celoss = nn.CrossEntropyLoss()
bceloss = nn.BCEWithLogitsLoss()
l1loss = nn.L1Loss(reduction = 'none')


g1 =  G().to(device)
g2 =  G().to(device)
c1 =  C().to(device)
c2 =  C().to(device)
d = D().to(device)
def high(x):
    return bceloss(x, torch.ones_like(x))

def low(x):
    return bceloss(x, torch.zeros_like(x))

def compare(x, y):
    #return (torch.softmax(x, 1) - torch.softmax(y, 1)).abs().mean()
    mx = torch.argmax(x, 1)
    mx = F.one_hot(mx, 10)
    my = torch.argmax(y, 1)
    my = F.one_hot(my, 10)
    return l1loss(x.softmax(1), y.softmax(1))
    
def entropy(x):
    return D(x).sigmoid().mean()

    

c_opt = optim.SGD(list(c1.parameters()) + list(c2.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
g_opt = optim.SGD(list(g1.parameters()) + list(g2.parameters()), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)


img_list = []
j = 0

source_enum = get_source(train = True, cycle = True)
target_enum = get_target(train = True, cycle = True)

for _ in range(1000):
    source_accs = []
    target_accs = []
    pbar = tqdm.tqdm(range(1000))
    for i in pbar:
        _, (source_x, true_source_c) = next(source_enum)
        _, (target_x, true_target_c) = next(target_enum)
        source_x = source_x.to(device)
        true_source_c = true_source_c.to(device)
        target_x = target_x.to(device)
        true_target_c = true_target_c.to(device)
        
        
        res = ""
        
        for _ in range(1):
            c_opt.zero_grad()
            g_opt.zero_grad()
            source_c1 = c1(g1(source_x))
            loss = celoss(source_c1, true_source_c)
            loss.backward()
            c_opt.step()
            g_opt.step()
        
        res += ' CLASS LOSS: ' + str(float(loss))
        
        for _ in range(a):
            c_opt.zero_grad()
            source_c1 = c1(g1(source_x))
            loss = celoss(source_c1, true_source_c)
            target_c1 = c1(g2(target_x))
            discrep_loss = entropy(target_c1)
            loss -= discrep_loss
            loss.backward()
            c_opt.step()
            
        res += ' DISCREP LOSS: ' + str(float(discrep_loss))
        
        for _ in range(b):
            g_opt.zero_grad()
            #source_c1, source_c2 = c1(g(source_x)), c2(g(source_x))
            #loss = celoss(source_c1, true_source_c) + celoss(source_c2, true_source_c)
            target_c1= c1(g2(target_x))
            discrep_loss = entropy(target_c1)
            loss = discrep_loss
            loss.backward()
            g_opt.step()
        
        max_values = torch.argmax(c1(g2(target_x)), 1)
        acc = torch.sum(max_values == true_target_c).float() / max_values.shape[0]
        res += ' ACC: ' + str(float(acc))
        
        j += 1
        pbar.set_description(res)
    
    nocycle_enum = get_target(train = True, cycle = False)
    test_enum =  get_target(train = False, cycle = False)
    
    accs = []
    for _, (real_target_x, true_target_y) in tqdm.tqdm(test_enum):
        real_target_x = real_target_x.to(device)
        true_target_y = true_target_y.to(device)
        real_target_c = c1(g2(real_target_x))
        max_values = torch.argmax(real_target_c, 1)
        acc = torch.sum(max_values == true_target_y).float() / max_values.shape[0]
        accs.append(acc)

    print(sum(accs) / len(accs))

        

        
    