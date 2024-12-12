import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
import sys
import torchvision
torch.autograd.set_detect_anomaly(True)
import argparse
from torch.utils.data import DataLoader as ImageLoader
import  random
from gcnnet import QKNET
from loss import CosLoss
from gcnload3 import get_images,get_testimages
from loss import TripletLossCosine as TripLoss3
# from loss import CosLoss,TripletLoss, HardNegativeContrastiveLoss,TripLoss3
from gcnload42 import Modelbou,get_dataset
from gcnnet44 import LSTMFE
from gcnnet45 import LSTMCLS

class QueryDataset(Dataset):
    def __init__(self,embs,batch_group,data_label):
        self.embs=embs
        self.batch_group=batch_group
        self.data_label=data_label
        self.data_list=self.construct_graph()
    def construct_graph(self):
        embs=self.embs.view(self.batch_group,-1,4,1000)
        data_list=[]
        for i in range(self.batch_group):
            avgfeature=embs[i] # 将向量转换为张量形式并进行标准化处理
            edge_index=torch.tensor([[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3], [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]], dtype=torch.long)
            label=self.data_label[i]
            data = Data(
                x=avgfeature,
                edge_index=edge_index,
                y=label
            )
            data_list.append(data)

        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class Retrieval:

    def __init__(self, args):
        self.args = args
        self.device=args.device

    def gcntrain(self):
        self.train()
        sys.exit()

    def shuffle_arrays(self,array1, array2):
        indices = list(range(len(array1))) # 创建一个随机顺序的索引列表
        random.shuffle(indices)# 使用索引列表来打乱两个数组
        shuffled_array1 = [array1[i] for i in indices]
        shuffled_array2 = [array2[i] for i in indices]
        return shuffled_array1, shuffled_array2

    def construct_namenum(self):
        name_num = {}
        num_name = {}
        num_train = 0
        datasetspath = self.args.feapath
        datasets = os.listdir(datasetspath)
        taskdict={}
        for dataset in datasets:
            taskname=dataset[4:]
            if taskname in list(taskdict.keys()):
                taskdict[taskname]=taskdict[taskname]+1
            else:
                taskdict[taskname]=1
        ttasks=list(taskdict.keys())
        for taskname in ttasks:
            if not taskname in list(name_num.keys()):
                name_num[taskname] = num_train
                num_name[num_train] = taskname
                num_train = num_train + 1
        classnum=len(list(name_num.values()))
        print(name_num.keys(),name_num.values(),"name_num")
        print(num_name.keys(),num_name.values(),"num_name")
        return  name_num, ttasks,num_name


    def train(self):

        self.name_num, ttasks, self.num_name =self.construct_namenum()
        classnum=len(list(self.name_num.values()))
        feaset11=Modelbou(self.args,self.args.feapath3,self.name_num)
        print("done loading features")
        self.qnum=self.args.qdatanum
        namelist=list(self.name_num.keys())
        nclass=len(namelist)
       
        self.mknet = LSTMFE(3*64*64, self.args.hidden, 2,nclass,self.args.embsize).to(self.device)
        self.mkclsnet=LSTMCLS(nclass,self.args.embsize).to(self.device)
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load(self.args.respath))
        self.resnet.to(self.device)
        self.lossf= TripLoss3()
        self.lossf2=torch.nn.CrossEntropyLoss()
        self.qklnet = QKNET(1000, 1000, 2,nclass,self.args.embsize).to(self.device)
        self.qklnet.to(self.device)
        self.mknet.to(self.device)
        self.mkclsnet.to(self.device)
        self.qklnet.train()
        self.mknet.train()
        self.mkclsnet.train()
        self.trainimgloader,self.testimgloader=get_images(self.args, self.num_name,self.name_num)
        print("done loading images")
        self.qnum=self.args.qdatanum
        print("done loading features")
        optimizer = torch.optim.Adam([
        dict(params=self.qklnet.parameters()),dict(params=self.mknet.parameters()),dict(params=self.mkclsnet.parameters())
        ], lr=self.args.lr)  # Only perform weight-decay on first convolution.
            # ], lr=self.args.lr,weight_decay=self.args.weight_decay)  # Only perform weight-decay on first convolution.
        print(len(self.trainimgloader),len(self.testimgloader),"len_train_loader")

        for curr_epoch in range(self.args.n_epochs):
            posnum=0
            for imgset in self.trainimgloader:
                embs,imglabel=imgset
                feas,nfeas=feaset11.get_query(imglabel)
                embs=embs.to(self.device)
                imglabel=imglabel.to(self.device)
                batch_group=embs.shape[0]
                qembs=self.qklnet(embs.to(self.device))
                #get model embeddings
                tfealoader = DataLoader(feas, batch_size=len(feas), shuffle=False,drop_last=False)
                for feaset in  tfealoader:
                    feaset=feaset.to(self.device)
                    membs,outx=self.mknet(feaset.to(self.device))
                pre_mlabel=self.mkclsnet(outx)
                # print(imglabel.shape,imglabel.shape,"img_label")
                loss1=self.lossf(qembs,membs,imglabel)
                loss2=self.lossf2(pre_mlabel,imglabel)
                loss=loss1+loss2
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            if curr_epoch%5==0:
                tcout=0
                tnum=0
                for imgset in self.testimgloader:
                    embs,imglabel=imgset
                    embs=embs.to(self.device)
                    imglabel=imglabel.to(self.device)
                    qembs=self.qklnet(embs.to(self.device))
                    #get model embeddings
                    feas,nfeas=feaset11.get_query(imglabel)
                    tfealoader = DataLoader(feas, batch_size=len(feas), shuffle=False,drop_last=False)
                    for feaset in  tfealoader:
                        feaset=feaset.to(self.device)
                        membs,_=self.mknet(feaset.to(self.device))
                    scores = torch.mm(membs, qembs.t())  # (160, 160)
                    sorted_model, max_ind = torch.sort(scores, 1, descending=True)
                    max_m = max_ind[:, :1]  # (160, 1)
                    for i in range(len(max_m)):
                        if imglabel[i]==imglabel[max_m[i,0]]:
                            tcout=tcout+1
                    tnum=tnum+embs.shape[0]
                print(tcout,loss.item(),"correct_count",tnum,"totnum")
                saved={'qknet':self.qklnet.state_dict(),'mknet':self.mknet.state_dict(),'mkclsnet':self.mkclsnet.state_dict()}
                torch.save(saved,'./models/'+str(curr_epoch)+'lcos_emb256_1e6_saved_models.pth')

    def test(self):
        self.name_num, ttasks, self.num_name =self.construct_namenum()
        classnum=len(list(self.name_num.values()))
        feaset11=Modelbou(self.args,self.args.feapath3,self.name_num)
        print("done loading features")
        self.qnum=self.args.qdatanum
        namelist=list(self.name_num.keys())
        nclass=len(namelist)
        modelpath='./models/'+'9000lcos_emb5_saved_models.pth'
        # modelpath='./models/'+str(7000)+'saved_models.pth'
        welltrainedmodels=torch.load(modelpath)
        self.mknet = LSTMFE(3*64*64, self.args.hidden, 2,nclass,self.args.embsize).to(self.device)
        self.mknet.eval()
        self.mkclsnet=LSTMCLS(nclass,self.args.embsize).to(self.device)
        self.mkclsnet.eval()
        # self.lossf= TripLoss3(args=self.args)
        # self.lossf2=torch.nn.CrossEntropyLoss()
        self.qklnet = QKNET(1000, 1000, 2,nclass,self.args.embsize).to(self.device)
        self.qklnet.eval()
        self.qklnet.load_state_dict(welltrainedmodels['qknet'])
        self.mknet.load_state_dict(welltrainedmodels['mknet'])
        self.mkclsnet.load_state_dict(welltrainedmodels['mkclsnet'])
        self.qklnet.to(self.device)
        self.mknet.to(self.device)
        self.mkclsnet.to(self.device)
        self.testimgloader=get_testimages(self.args, self.num_name,self.name_num)
        #测试实验的时候用
        # _,self.testimgloader=get_images(self.args, self.num_name,self.name_num)
        print("done loading images")
        self.qnum=self.args.qdatanum
        tcout=0
        tnum=0
        with torch.no_grad():
            for imgset in self.testimgloader:
                embs,imglabel=imgset
                embs=embs.to(self.device)
                imglabel=imglabel.to(self.device)
                qembs=self.qklnet(embs.to(self.device))
                #get model embeddings
                feas,nfeas=feaset11.get_query(imglabel)
                tfealoader = DataLoader(feas, batch_size=len(feas), shuffle=False,drop_last=False)
                for feaset in  tfealoader:
                    feaset=feaset.to(self.device)
                    membs,_=self.mknet(feaset.to(self.device))
                scores = torch.mm(membs, qembs.t())  # (160, 160)
                sorted_model, max_ind = torch.sort(scores, 1, descending=True)
                max_m = max_ind[:, :1]  # (160, 1)
                for i in range(len(max_m)):
                    if imglabel[i]==imglabel[max_m[i,0]]:
                        tcout=tcout+1
                tnum=tnum+embs.shape[0]
            print(tcout,"correct_count",tnum,"totnum")      





class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def str2bool(self, s):
        return s.lower() in ['true', 't']

    def set_arguments(self):
        ##############################################
        self.parser.add_argument('--device', type=str, default='cuda:0', help='gpus to use, i.e. 0')
        self.parser.add_argument('--mode', type=str, default='train', help='i.e. train, test')
        self.parser.add_argument('--seed', type=int, default=888, help='seed for reproducibility')
        self.parser.add_argument('--batch-size', type=int, default=200, help='batch size')
        self.parser.add_argument('--testbatchsize', type=int, default=100, help='batch size')
        self.parser.add_argument('--qdatanum', type=int, default=5, help='batch size') #一个类有多少图像
        self.parser.add_argument('--n_epochs', type=int, default=10000, help='dimension of embedding')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        ###############################################
        self.parser.add_argument('--modelpath', type=str, default='./../../../SZYretrival/szyretrival/data/mobilemodels')
        self.parser.add_argument('--testimagepath', type=str, default='./../../../SZYretrival/szyretrival/data/tar/mnrtest')
        self.parser.add_argument('--trainimagepath', type=str, default='./../../../SZYretrival/szyretrival/data/tar/mnrtrain')
        self.parser.add_argument('--respath', type=str, default='./../../data/res18.pth')
        #################################################
        self.parser.add_argument('--feapath3', type=str, default='./../../data/modelbou5')
        self.parser.add_argument('--feapath', type=str, default='./../../data/modelbou6')
        self.parser.add_argument('--num_class', type=int, default=4)
        ###################################################GATargs
        self.parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters).')
        self.parser.add_argument('--hidden', type=int, default=500, help='Number of hidden units.')
        self.parser.add_argument('--embsize', type=int, default=256, help='Number of hidden units.')
        self.parser.add_argument('--nheads', type=int, default=3, help='Number of head attentions.')
        self.parser.add_argument('--dropout', type=float, default=0.7, help='Dropout rate (1 - keep probability).')
        self.parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
        self.parser.add_argument('--patience', type=int, default=100, help='Patience')
        self.parser.add_argument('--nclass', type=int, default=100, help='Patience')

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args

if __name__ == '__main__':
    parse=Parser().parse()
    Retrieval(parse).gcntrain()
    # Retrieval(parse).test()
