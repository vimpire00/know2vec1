import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from torch.utils.data import Dataset,random_split
from torch_geometric.data import DataLoader
# from torch_geometric.data.dataloader import DataLoader
from torch.utils.data import DataLoader as ImageLoader
import torchvision
import  random
import argparse
from argpar import Parser
import gc
import random
from gcnnet import Mobilenet

class Modelbou(Dataset):  #Modelbou(args,negpath,name_num)
    def __init__(self, args,feapath,name_num):
        super(Modelbou, self).__init__()
        self.resize = 64
        self.args=args
        self.path=feapath
        self.datasets=os.listdir(self.path)
        self.name_num=name_num
        self.device=args.device
        self.qnum=args.qdatanum
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load(self.args.respath))
        self.resnet=self.resnet.to(args.device)
        self.resnet.eval()
        self.data_list=self.loaddata()
        print(len(self.data_list),"len_data_list")
        del self.resnet
 
    def __len__(self):
        return len(self.data_list)

    def cosdis(self,emb1,emb2):
        l2f=torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        x=l2f(emb1,emb2)
        return x[0]

    def loaddata(self):
        data_list={}
        # data_list=[[]]
        # flag=0
        i=0
        for i in range(58):
            data_list[i]=[]
        j=0
        for dataset in self.datasets:
            j=j+1
            # if j>1000:
            #     break
            taskname=dataset[4:]
            if not taskname in list(self.name_num.keys()):
                continue
            cdatapath=os.path.join(self.path,dataset)
            cdata=torch.load(cdatapath)
            adv_list=cdata['adv']#[4,4,3,64,64]
            nodes=cdata['node']#[4,3,64,64]
            dataname=cdata['dataname']
            taskname=cdata['taskname']
            featurelabel=self.name_num[taskname]
            with torch.no_grad():
                currm=[]
                nodefeature=nodes.to(self.device)
                features=adv_list.to(self.device).reshape(-1,3,64,64)
                for i in range(4):
                    features[i*4+i]=nodefeature[i]
                edge_index = torch.tensor([[0, 0, 0,0,1,1,1,1,2,2,2,2,3,3,3,3], [0,1, 2, 3,0,1, 2, 3,0,1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)  # 第一个点到其他三个点的连接关系
                features=self.resnet(features)
                data = Data(
                    x=nodefeature,
                    edge_index=edge_index,  
                    edge_attr=features,
                    y=torch.tensor(featurelabel)
                )
                currm.append(data)
            data_list[featurelabel].extend(currm)
        return data_list

    def split(self):
        names=list(self.data_list.keys())
        print(names,"names",len(names),"len_names")
        train_list=[]
        test_list=[]
        for name in names:
            llen=len(self.data_list[name])
            data=self.data_list[name]
            train_list.extend(data)
        test_list=train_list
        return train_list,test_list


    def get_query(self,labels):
        datas=[]
        ndatas=[]
        for i in labels:
            max_value=len(self.data_list[i.item()])
            random_integer = random.randint(0, max_value-1)
            # print(len(self.data_list[i.item()]),random_integer,"len_self_data_list")
            datas.append(self.data_list[i.item()][random_integer])
            ndatas.append(self.data_list[(i.item()+1)%len(self.data_list)][random_integer%len(self.data_list[(i.item()+1)%len(self.data_list)])] )
        # datas=[self.data_list[i][] for i in labels]
        # ndatas=[self.data_list[(i+1)%len(self.data_list)] for i in labels]

        return datas,ndatas
    def __getitem__(self, index):
        len_data=len(self.data_list)
        trfea = self.data_list[index]  #[numgroup,qnum,4,3,64,64]
        return trfea


def get_feature(args,name_num):

    feaspath=args.feapath3
    datasets=os.listdir(feaspath)
    feadata=Modelbou(args,feaspath,name_num)
    print(len(feadata),"len_total_datas")
    # 创建包含所有样本的新数据集
    num_total_graphs = len(feadata)
    indices = list(range(num_total_graphs))
    random.shuffle(indices)
    num_train_graphs = 64
    train_indices = indices[:-num_train_graphs]
    test_indices = indices[-num_train_graphs:]
    # Convert indices to integers
    train_indices = [int(idx) for idx in train_indices]
    test_indices = [int(idx) for idx in test_indices]
    train_list = [feadata[idx] for idx in train_indices]
    test_list = [feadata[idx] for idx in test_indices]
    # train_list, test_list = train_test_split_edges(feadata,test_ratio=0.2)
    trainfealoader = DataLoader(train_list, batch_size=args.batch_size, shuffle=True,drop_last=False)
    testfealoader = DataLoader(test_list, batch_size=args.batch_size, shuffle=True,drop_last=False)
    print(len(train_list),len(test_list),"len_train_test_list")
    return trainfealoader,testfealoader


def get_dataset(args,name_num):

    feaspath=args.feapath
    datasets=os.listdir(feaspath)
    feadata=Modelbou(args,feaspath,name_num)
    print(len(feadata),"len_total_datas")
    
    return feadata



if __name__ == '__main__':
    args=Parser().parse()
    # get_dataset()
    print(args.feapath,"path")
    