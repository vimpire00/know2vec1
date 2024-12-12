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
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from torch_geometric.utils import to_dense_adj
import torch.nn as nn

class QKNET(nn.Module):
    def __init__(self, input_channels, hidden_size, num_layers,totclass,embsize):
        super(QKNET, self).__init__()
        self.totclass=totclass
        self.embsize=embsize
        self.hidden_size = hidden_size
        self.num_layers = num_layers*2
        self.lstm = nn.LSTM(input_channels,self.hidden_size, 1, batch_first=True,bidirectional=True)
        self.classf1=nn.Linear(self.hidden_size*2,self.embsize)
        # self.connet=nn.Linear(1000*4,self.embsize)
        # self.avgnet=nn.Linear(1000,self.embsize)
        self.sigmoid=torch.nn.LeakyReLU()
    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=-1, keepdim=True)
        x = torch.div(x, norm2)
        return x
    def forward(self, data):#LSTM
        x=data
        # x=self.l2norm(x)
        x = x.view(-1,4,x.shape[-1])  # 将图像展平为一维向量
        # print(x.shape,"shape55")
        h0 = torch.randn(2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.randn(2, x.size(0), self.hidden_size).to(x.device)
        out, att0 = self.lstm(x, (h0, c0))  # 输出out的形状为(batch_size, sequence_length, hidden_size)
        out=self.classf1(out[:,-1,:])
        out=self.l2norm(out)
        return out
    # def forward(self, data):#CONNET
    #     x=data
    #     # x=self.l2norm(x)
    #     x = x.view(-1,4,x.shape[-1])  # 将图像展平为一维向量
    #     out=self.connet(x.view(x.shape[0],-1))
    #     out=self.l2norm(out)
    #     return out
    # def forward(self, data):#AVG
    #     x=data
    #     # x=self.l2norm(x)
    #     x = x.view(-1,4,x.shape[-1])  # 将图像展平为一维向量
    #     x=torch.mean(x,dim=1)
    #     out=self.avgnet(x)
    #     # out=self.connet(x.view(x.shape[0],-1))
    #     out=self.l2norm(out)
    #     return out


class Mobilenet(torch.nn.Module):
    def __init__(self, args):
        super(Mobilenet, self).__init__()
        self.args = args
        pretrained_mobnet = torchvision.models.mobilenet_v2(pretrained=False)
        self.fea_exa = pretrained_mobnet
        self.model_fc1 = torch.nn.Sequential()
        self.model_fc1.add_module('model_fc', torch.nn.Linear(1000,4))
        self.softmax=torch.nn.Softmax(dim=1)

    def forward(self,x):
        x=self.fea_exa(x)
        x=self.model_fc1(x)
        x = self.softmax(x)

        return x