import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset,random_split
import sys
import torchvision
torch.autograd.set_detect_anomaly(True)
import argparse
from torch.utils.data import DataLoader as ImageLoader
import random
from he_nets2 import LSTMFE,LSTMCLS,QNET
from hete_loader2 import get_datasets_emb
from hete_qkloader2 import  LearnwareDataset
# from loss import TripletLossCosine1 as TripLoss3
# from loss import TripLoss3_HETE as TripLoss3
from  loss import TripLoss3_HETE_Cosine as TripLoss3
import scipy.stats
import numpy as np

def measure_test(outputs, labels, only_core=True):
    m_results = {}
    m_results['weightedtau'] = scipy.stats.weightedtau(labels, outputs, rank=None).correlation
    m_results['pearsonr'] = scipy.stats.pearsonr(outputs, labels)[0]
    m_results['spearman'] = scipy.stats.spearmanr(outputs, labels)[0]

    if only_core:
        return m_results


class Retrieval:

    def __init__(self, args):
        self.args = args
        self.device=args.device
        self.embsize=128
        self.hidden_size=128
    def gcntrain(self):
        self.train()
        # self.test_mlabel()
        sys.exit()

    def shuffle_arrays(self,array1, array2):
        indices = list(range(len(array1))) # 创建一个随机顺序的索引列表
        random.shuffle(indices)# 使用索引列表来打乱两个数组
        shuffled_array1 = [array1[i] for i in indices]
        shuffled_array2 = [array2[i] for i in indices]
        return shuffled_array1, shuffled_array2

    def train(self):
        # self.feature_dict=get_datasets_emb()
        self.memb_dataset=get_datasets_emb()
        # modelnum=self.feature_dict.get_modelnum()
        modelnum=self.args.num_learnware  ##在每个模型只有1个Emb时失效
        #LSTMinput_channels, hidden_size, num_layers
        #  input_channels, hidden_size, embsize,num_layers, modelnum)
        self.mknet = LSTMFE(1000, self.hidden_size,self.embsize, 2,modelnum).to(self.device)
        self.mknet.train()
        self.mkclsnet=LSTMCLS(self.embsize, modelnum).to(self.device)
        self.mkclsnet.train()
        self.qknet= QNET(1000, self.hidden_size, self.embsize,2).to(self.device)
        self.qknet.train()
        self.lossf= TripLoss3(self.args)
        self.lossf2=torch.nn.CrossEntropyLoss()
        print("done loading images")
        print("done loading features")
        optimizer = torch.optim.Adam([ dict(params=self.qknet.parameters()),
        dict(params=self.mknet.parameters())
        ], lr=self.args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # gamma是学习率衰减率

        # optimizer = torch.optim.Adam([ dict(params=self.qknet.parameters()),
        # dict(params=self.mkclsnet.parameters()),dict(params=self.mknet.parameters())
        # ], lr=self.args.lr)
        # Only perform weight-decay on first convolution.
        # 使用random_split函数分割数据集
        train_dataset = LearnwareDataset(args=self.args, stype='train', heterogeneous=True)
        split=10
        num_samples=len(train_dataset)
        train_subset, val_subset = random_split(train_dataset, [num_samples - split,split])
        # 创建训练数据加载器
        self.data_loader_train = DataLoader(
            train_subset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True,
            collate_fn=LearnwareDataset.collate_fn
        )
        # 创建验证数据加载器
        self.data_loader_val = DataLoader(
            val_subset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,  # 通常不需要在验证数据加载器中进行shuffle
            collate_fn=LearnwareDataset.collate_fn
        )

        test_dataset = LearnwareDataset(args=self.args, stype='test',
                                         heterogeneous=True)
        # 创建验证数据加载器
        self.data_loader_test = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,  # 通常不需要在验证数据加载器中进行shuffle
            collate_fn=LearnwareDataset.collate_fn
        )
        self.memb_dataset_loader,self.memb_dataset=get_datasets_emb()
        self.test()
        for curr_epoch in range(self.args.n_epochs):
            self.qknet.train()
            self.mknet.train()
            for step, (inputs, labels,padlength) in enumerate(self.data_loader_train):
                labels = labels.to(device=self.args.device)
                # Find the length of the shortest list
                min_length = min(len(lst) for lst in inputs)
                # Truncate each list to the length of the shortest list
                truncated_data = [lst[:min_length] for lst in inputs]
                qembs=self.qknet(truncated_data,padlength)
                membss, mlabels = self.memb_dataset.get_query(curr_epoch)
                mlabels=torch.tensor(mlabels).to(device=self.args.device)
                membs,outemb2label=self.mknet(membss)
                # print(membs.shape,outemb2label.shape,"outemb")
                # pre_mlabel=self.mkclsnet(outemb2label)
                #get model embeddings
                loss1=self.lossf(membs,qembs,labels)
                # print(pre_mlabel.shape,labels.shape,"labshape")
                # loss2=self.lossf2(pre_mlabel,mlabels)
                loss=loss1
                # loss1=self.lossf(membs,qembs,labels)
                print(loss.item(),"loss_item")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            self.test()
            if curr_epoch%20==0:
                model_dict={}
                model_dict['mknet']=self.mknet.state_dict()
                model_dict['qknet']=self.qknet.state_dict()
                torch.save(model_dict,'./framwork/'+'heteregreinc1e4'+str(curr_epoch)+'.pth')
            # with torch.no_grad():
            #     self.qknet.eval()
            #     self.mknet.eval()
            #     # self.mkclsnet.eval()            
            #     for step, (inputs, labels,padlength) in enumerate(self.data_loader_val):
            #         labels = labels.to(device=self.args.device)
            #         # Find the length of the shortest list
            #         min_length = min(len(lst) for lst in inputs)
            #         # Truncate each list to the length of the shortest list
            #         truncated_data = [lst[:min_length] for lst in inputs]
            #         qembs=self.qknet(truncated_data,padlength)
            #         membss, mlabels = self.memb_dataset.get_query(curr_epoch)
            #         mlabels=torch.tensor(mlabels).to(device=self.args.device)
            #         membs,outemb2label=self.mknet(membss)
            #         # prelabel=torch.argmax(outlabel,dim=1)
            #         scores = torch.mm(membs, qembs.t()) # (44, 160)
            #         scores=scores.t()
            #         # print(labels.shape,mlabels.shape,"shape333")
            #         positive_mask = (mlabels.unsqueeze(0) == labels.unsqueeze(1))
            #         positive_indices = positive_mask.nonzero(as_tuple=True)
            #         sorted_scores,sorted_indices=torch.sort(scores,descending=True)
            #         # pre_top=sorted_indices
            #         # print(sorted_indices.shape,labels.shape,"bbbb")
            #         outputs = np.asarray(sorted_indices.cpu(), dtype=np.float32)
            #         labels = np.asarray(labels.cpu(), dtype=np.float32)
            #         meascore=measure_test(outputs,labels)
            #         print(outputs[0],labels[0],meascore,"val_meascore")
                    # print(pre_top,positive_indices[1],"indicee")
                    # target=pre_top.eq(positive_indices[1])
            # mlabels = torch.tensor([i for i in range(self.args.num_learnware)]).to(self.device)


    def test(self):
        # mlabels = torch.tensor([i for i in range(self.args.num_learnware)]).to(self.device)
        tnum=0
        curr_epoch=0
        for step, (inputs, labels,qlength) in enumerate(self.data_loader_test):
            labels=torch.stack(labels)
            labels = labels.to(device=self.args.device)
            inputs=inputs.to(device=self.args.device)
            qembs=self.qknet(inputs,qlength)
            membss, mlabels = self.memb_dataset.get_query(curr_epoch)
            mlabels=torch.tensor(mlabels).to(device=self.args.device)
            membs,outemb2label=self.mknet(membss)
            scores = torch.mm(qembs, membs.t())  # (160, 160)
            labels=labels.t()
            kscores=[]
            for i in range(scores.shape[0]):
                # label=100-labels[i]
                label=labels[i]
                # print(label.shape,"label_shape")
                non_zero_indices = (label != 0)
                labell = label[non_zero_indices].detach().cpu().numpy()
                # print(labell,"labell")
                outputt = scores[i][non_zero_indices].detach().cpu().numpy()
                min_val = np.min(outputt)  # 计算最大值和最小值
                max_val = np.max(outputt)
                normalized_out = (outputt - min_val) / (max_val - min_val)    # 归一化
                min_val = np.min(labell)  # 计算最大值和最小值
                max_val = np.max(labell)
                normalized_label = (labell - min_val) / (max_val - min_val)    # 归一化
                score=measure_test(normalized_out,normalized_label)
                # score=measure_test(normalized_out,normalized_label)
                kscores.append(score)  
                if i%2==0:
                    print(score,i,"sscore")
            # print(kscores[0],kscores[2],kscores[4],kscores[6],kscores[8],"k-score")

    def test1(self):  #直接运行，而不是在训练的时候测试，先加载模型再测试
        self.memb_dataset=get_datasets_emb()
        modelnum=self.args.num_learnware  ##在每个模型只有1个Emb时失效
        #LSTMinput_channels, hidden_size, num_layers
            #  input_channels, hidden_size, embsize,num_layers, modelnum)
        self.mknet = LSTMFE(1000, self.hidden_size,self.embsize, 2,modelnum).to(self.device)
        self.mknet.eval()
        self.qknet= QNET(1000, self.hidden_size, self.embsize,2).to(self.device)
        self.qknet.eval()
        model_dict=torch.load('./framwork/'+'heteregreinc1e4160.pth',map_location='cpu')
        self.mknet.load_state_dict(model_dict['mknet'])
        self.qknet.load_state_dict(model_dict['qknet'])
        test_dataset = LearnwareDataset(args=self.args, stype='test',
                                         heterogeneous=True)
        # 创建验证数据加载器
        self.data_loader_test = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,  # 通常不需要在验证数据加载器中进行shuffle
            collate_fn=LearnwareDataset.collate_fn
        )
        self.memb_dataset_loader,self.memb_dataset=get_datasets_emb()
        self.test()



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
        self.parser.add_argument('--num_workers', type=int, default=0, help='dimension of embedding')
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        ###############################################
        # self.parser.add_argument('--modelpath', type=str, default='./../../../SZYretrival/szyretrival/data/mobilemodels')
        # self.parser.add_argument('--testimagepath', type=str, default='./../../../SZYretrival/szyretrival/data/tar/mnrtest')
        # self.parser.add_argument('--trainimagepath', type=str, default='./../../../SZYretrival/szyretrival/data/tar/mnrtrain')
        # self.parser.add_argument('--respath', type=str, default='./../../data/res18.pth')
        # #################################################
        # self.parser.add_argument('--feapath3', type=str, default='./../../data/modelbou5')
        # self.parser.add_argument('--feapath', type=str, default='./../../data/modelbou6')
        # self.parser.add_argument('--num_class', type=int, default=4)
        # './../../ModelSpider/tools/hetetraintaskrank.npy'  'train_dataset_path
        self.parser.add_argument('--train_dataset_path', type=str, default='./../../data/szyheteres18trainemb')
        # self.parser.add_argument('--train_dataset_path', type=str, default='./../../data/heres18emb')
        self.parser.add_argument('--test_dataset_path',type=str, default='./../../data/testres18emb')
        self.parser.add_argument('--test_size_threshold', type=int, default=1536 )
        self.parser.add_argument('--data_url', type=str, default='./../../data/szyheteres18trainemb')
        self.parser.add_argument('--heterogeneous_sampled_minnum', type=int, default=4)
        self.parser.add_argument('--heterogeneous_sampled_maxnum', type=str, default=4)
        self.parser.add_argument('--num_prototypes', type=int, default=None)
        self.parser.add_argument('--num_learnware', type=int, default=48)  #候选模型数量
        self.parser.add_argument('--fixed_gt_size_threshold', type=int, default=128)
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
    # Retrieval(parse).gcntrain()
    Retrieval(parse).test1()
