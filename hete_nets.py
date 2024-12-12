import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision

class Mobilenet(torch.nn.Module):

    def __init__(self, dim1,dim2):
        super(Mobilenet, self).__init__()
        pretrained_mobnet = torchvision.models.mobilenet_v2(pretrained=False)
        self.fea_exa = pretrained_mobnet
        self.model_fc1 = torch.nn.Sequential()
        self.model_fc1.add_module('model_fc', torch.nn.Linear(dim1,dim2))
        self.softmax=torch.nn.Softmax(dim=1)

    def forward(self,x):
        x=self.fea_exa(x)
        x=self.model_fc1(x) 
        x = self.softmax(x)

        return x


class DenseNet(torch.nn.Module):

    def __init__(self, dim1,dim2):
        super(DenseNet, self).__init__()
        # self.args = args
        pretrained_mobnet = torchvision.models.densenet201(pretrained=False)
        self.module = pretrained_mobnet
        self.module.add_module('classifier', torch.nn.Linear(dim1,dim2))
    def forward(self,x):
        x=self.module(x)
        # print(x.shape,"shape55")

        return x


class Inception_v3(torch.nn.Module):

    def __init__(self, dim1,dim2):
        super(Inception_v3, self).__init__()
        self.module =torchvision.models.inception_v3(pretrained=False)
        self.module.add_module('fc', torch.nn.Linear(dim1,dim2))
    def forward(self,x):
        x,x1=self.module(x)

        return x

class Res50(torch.nn.Module):

    def __init__(self, dim1,dim2):
        super(Res50, self).__init__()
        self.module =torchvision.models.resnet50(pretrained=False)
        self.module.add_module('fc', torch.nn.Linear(dim1,dim2))
    def forward(self,x):
        x=self.module(x)

        return x
