import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch.utils.data import Dataset,random_split
from torch_geometric.data import DataLoader
# from torch_geometric.data.dataloader import DataLoader
from torch.utils.data import DataLoader as ImageLoader
import torchvision
import  random
import argparse
from argpar import Parser
# import gc

class TrainImage(Dataset):
    def __init__(self, args,imgpath,label):
        super(TrainImage, self).__init__()
        self.resize = 64
        self.path =imgpath
        self.args=args
        self.device=args.device
        self.qnum=args.qdatanum
        self.data = torch.load(self.path)
        self.keys=list(self.data.keys())
        self.conflabel=label
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load(self.args.respath))
        self.resnet=self.resnet.to(args.device)
        self.resnet.eval()
        self.embs = self.loaddata(self.data)
        self.embs.cpu()
        torch.cuda.empty_cache()
        del self.resnet
        # del self.embs

    def __len__(self):
        return len(self.embs)

    def loaddata(self,data):
        cls1,cls2,cls3,cls4=data[self.keys[0]],data[self.keys[1]],data[self.keys[2]],data[self.keys[3]]
        len=cls1.shape[0]
        labels=torch.tensor([0,1,2,3])
        imgs=torch.cat([cls1.unsqueeze(0),cls2.unsqueeze(0),cls3.unsqueeze(0),cls4.unsqueeze(0)],0)
        imgs=imgs.transpose(0,1)#torch.Size([20, 4, 3, 64, 64])
        totalgroup=imgs.shape[0]//self.qnum
        imgs=imgs[:self.qnum*totalgroup,:,:,:,:]
        imgs=imgs.reshape(-1,self.qnum,4,3,64,64)
        embs=[]
        with torch.no_grad():
            for i in range(imgs.shape[0]):
                img=imgs[i]
                img=img.to(self.device)
                emb=self.resnet(img.reshape(-1,3,64,64))
                emb=emb.view(-1,4,1000)
                emb=torch.mean(emb,dim=0)
                embs.append(emb)
                del img
                del emb
        embs=torch.stack(embs).squeeze().cpu()
        if embs.dim() == 2:
            embs=embs.unsqueeze(0)
        torch.cuda.empty_cache()
        del imgs
        return embs

    def __getitem__(self, index):
        emb=self.embs[index]#[numgrou,4,1000]
        # imgs = self.images[index,:,:,:,:,:]#[numgroup,qnum,4,3,64,64]
        label=self.conflabel
        return emb,label


def get_images(args,num_name,name_num):
    datasetspath=args.trainimagepath
    datasets=os.listdir(datasetspath)
    print(datasets,"datasets")
    print(list(name_num.keys()),"keys")
    totalimages=[]
    for dataset in datasets:
        negpath=os.path.join(datasetspath,dataset)
        taskname=dataset+'h'
        if not taskname in list(name_num.keys()):
            continue
        else:
            print(taskname)
        num_train=name_num[taskname]
        negdataset = TrainImage(args,negpath,num_train)
        totalimages.append(negdataset)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        del negdataset
    random.shuffle(totalimages)# 将列表分成两个子列表
    trainimages=totalimages
    combined_dataset = torch.utils.data.ConcatDataset(trainimages)
    trainimgloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True,drop_last=False)
    datasetspath=args.testimagepath
    datasets=os.listdir(datasetspath)
    totalimages=[]
    for dataset in datasets:
        negpath=os.path.join(datasetspath,dataset)
        taskname=dataset+'h'
        if not taskname in list(name_num.keys()):
            continue
        num_train=name_num[taskname]
        negdataset = TrainImage(args,negpath,num_train)
        totalimages.append(negdataset)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        del negdataset
    random.shuffle(totalimages)# 将列表分成两个子列表
    combined_dataset = torch.utils.data.ConcatDataset(totalimages)
    print(len(combined_dataset ),"len_testimg")
    testimgloader = DataLoader(combined_dataset, batch_size=args.testbatchsize, shuffle=True,drop_last=False)
    # combined_testdataset = torch.utils.data.ConcatDataset(testimages)
    # testimgloader = DataLoader(combined_testdataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    return trainimgloader,testimgloader



def get_testimages(args,num_name,name_num):
    datasetspath=args.testimagepath
    datasets=os.listdir(datasetspath)
    totalimages=[]
    for dataset in datasets:
        negpath=os.path.join(datasetspath,dataset)
        taskname=dataset+'h'
        if not taskname in list(name_num.keys()):
            continue
        num_train=name_num[taskname]
        negdataset = TrainImage(args,negpath,num_train)
        totalimages.append(negdataset)
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        del negdataset
    random.shuffle(totalimages)# 将列表分成两个子列表
    combined_dataset = torch.utils.data.ConcatDataset(totalimages)
    print(len(combined_dataset ),"len_testimg")
    testimgloader = DataLoader(combined_dataset, batch_size=args.testbatchsize, shuffle=True,drop_last=False)
    return testimgloader




if __name__ == '__main__':
    args=Parser().parse()
    print(args.feapath,"path")
    name_num={}
    num_train=0
    datasetspath=args.testimagepath
    datasets=os.listdir(datasetspath)
    for dataset in datasets:
        name_num[dataset[:-3]]=num_train
        num_train=num_train+1
    get_feature(args,name_num)