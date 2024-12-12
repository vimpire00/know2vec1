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
    def __init__(self, input_channels, hidden_size, num_layers,totclass,embsize):
        super(LSTMFE, self).__init__()
        self.totclass=totclass
        self.hidden_size = hidden_size
        self.embsize=embsize
        self.num_layers = num_layers*2
        self.lstm0 = nn.LSTM(1000,self.hidden_size, 1, batch_first=True,bidirectional=True)
        self.lstm1 = nn.LSTM(1000,self.hidden_size, 1, batch_first=True,bidirectional=True)
        self.lstm2 = nn.LSTM(1000,self.hidden_size, 1, batch_first=True,bidirectional=True)
        self.lstm3 = nn.LSTM(1000,self.hidden_size, 1, batch_first=True,bidirectional=True)
        self.lstm4 = nn.LSTM(self.hidden_size*2,self.embsize, 1, batch_first=True,bidirectional=False)
        # self.classf1=nn.Linear(1000*8,256)
        self.classf1=nn.Linear(self.hidden_size*8,self.embsize)
        # self.classf1=nn.Linear(self.hidden_size*2,self.embsize)
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
        h0 = torch.randn(2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.randn(2, x.size(0), self.hidden_size).to(x.device)
        # 前向传播
        out0, att0 = self.lstm0(x[:,0,:,:], (h0, c0))  # 输出out的形状为(batch_size, sequence_length, hidden_size)
        out1, att1 = self.lstm1(x[:,1,:,:], (h0, c0))  # 输出out的形状为(batch_size, sequence_length, hidden_size)
        out2, att2 = self.lstm2(x[:,2,:,:], (h0, c0))  # 输出out的形状为(batch_size, sequence_length, hidden_size)
        out3, att3 = self.lstm3(x[:,3,:,:], (h0, c0))  # 输出out的形状为(batch_size, sequence_length, hidden_size)
        # outx=torch.cat([out0[:,-1,:],out1[:,-1,:],out2[:,-1,:],out3[:,-1,:]],dim=-1)connect
        # outx=self.classf1(outx)
        # out=self.l2norm(outx)
        # return out,outx
        outx=torch.cat([out0[:,-1,:].unsqueeze(1),out1[:,-1,:].unsqueeze(1),out2[:,-1,:].unsqueeze(1),out3[:,-1,:].unsqueeze(1)],dim=1)
        outx,_=self.lstm4(outx)
        out=self.l2norm(outx[:,-1,:])
        return out,outx[:,-1,:]
        #avg
        # outx=torch.cat([out0[:,-1,:].unsqueeze(1),out1[:,-1,:].unsqueeze(1),out2[:,-1,:].unsqueeze(1),out3[:,-1,:].unsqueeze(1)],dim=1)
        # outx=torch.mean(outx,dim=1).squeeze()
        # outx=self.classf1(outx)
        # out=self.l2norm(outx)
        # # #outx 用于分类
        # return out,outx
##lstm