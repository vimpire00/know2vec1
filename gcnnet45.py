import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch.utils.data import Dataset
import torchvision
import torch
import torch.nn as nn

class LSTMFE(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers,totclass):
        super(LSTMFE, self).__init__()
        self.totclass=totclass
        self.hidden_size = hidden_size
        self.num_layers = num_layers*2
        self.lstm0 = nn.LSTM(1000,self.hidden_size, 1, batch_first=True,bidirectional=True)
        self.lstm1 = nn.LSTM(1000,self.hidden_size, 1, batch_first=True,bidirectional=True)
        self.lstm2 = nn.LSTM(1000,self.hidden_size, 1, batch_first=True,bidirectional=True)
        self.lstm3 = nn.LSTM(1000,self.hidden_size, 1, batch_first=True,bidirectional=True)
        # self.lstm4 = nn.LSTM(1000*2,1000, 1, batch_first=True,bidirectional=True)
        # self.classf1=nn.Linear(1000*8,128)
        self.classf1=nn.Linear(1000*2,128)
        self.classf2=nn.Linear(128,self.totclass)
        self.sigmoid=torch.nn.LeakyReLU()
        self.softmax=torch.nn.Softmax(dim=1)
    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=-1, keepdim=True)
        x = torch.div(x, norm2)
        return x
    def forward(self, data):
        # x的形状为(batch_size, sequence_length, input_channels, height, width)
        x = data.edge_attr
        batch_size=x.shape[0]//16
        x = x.view(batch_size,4,4,-1)  # 将图像展平为一维向量
        # 初始化LSTM的隐藏状态
        # x=self.l2norm(x)
        x=self.l2norm(x)
        h0 = torch.randn(2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.randn(2, x.size(0), self.hidden_size).to(x.device)
        # 前向传播
        out0, att0 = self.lstm0(x[:,0,:,:], (h0, c0))  # 输出out的形状为(batch_size, sequence_length, hidden_size)
        out1, att1 = self.lstm1(x[:,1,:,:], (h0, c0))  # 输出out的形状为(batch_size, sequence_length, hidden_size)
        out2, att2 = self.lstm2(x[:,2,:,:], (h0, c0))  # 输出out的形状为(batch_size, sequence_length, hidden_size)
        out3, att3 = self.lstm3(x[:,3,:,:], (h0, c0))  # 输出out的形状为(batch_size, sequence_length, hidden_size)
        out=torch.cat([out0[:,-1,:].unsqueeze(1),out1[:,-1,:].unsqueeze(1),out2[:,-1,:].unsqueeze(1),out3[:,-1,:].unsqueeze(1)],dim=1)
        out=torch.mean(out,dim=1).squeeze()
        out=self.classf1(out)
        out=self.sigmoid(out)
        out=self.classf2(out)
        classn=self.softmax(out)
        return out


class LSTMCLS(nn.Module):
    def __init__(self, totclass,embsize):
        super(LSTMCLS, self).__init__()
        self.totclass=totclass
        self.classf1=nn.Linear(embsize,128)
        self.classf2=nn.Linear(128,self.totclass)
        self.sigmoid=torch.nn.LeakyReLU()
        self.softmax=torch.nn.Softmax(dim=1)
    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=-1, keepdim=True)
        x = torch.div(x, norm2)
        return x
    def forward(self, modelvec):
        # print(modelvec.shape,"shape666")
        out=self.classf1(modelvec)
        out=self.sigmoid(out)
        out=self.classf2(out)
        classn=self.softmax(out)
        return out