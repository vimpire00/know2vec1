import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import  random


class TrainImage(Dataset):
    def __init__(self,oripath,imgname,label):
        super(TrainImage, self).__init__()
        self.resize = 64
        self.path =os.path.join(oripath,imgname)
        self.respath='./../../data/res18.pth'
        self.device='cuda:0'
        self.data = torch.load(self.path)
        self.conflabel=label
        self.resnet = torchvision.models.resnet18(pretrained=False)
        self.resnet.load_state_dict(torch.load(self.respath,map_location='cpu'))
        self.resnet=self.resnet.to(self.device)
        self.resnet.eval()
        self.embs = self.loaddata(self.data)
        print(len(self.embs),"len_self_emb")
        torch.cuda.empty_cache()
        del self.resnet
        del self.data

    def __len__(self):
        return len(self.embs)

    def loaddata(self,data1): # 提取所有图像的特征
        adv_list=data1['adv']
        totalfeatures = [[] for _ in range(len(adv_list))]  # 初始化一个二维列表
        for i in range(len(adv_list)):
            class_list_features = adv_list[i]
            j=-1
            for class_list_features_j in class_list_features:
                j=j+1
                if  not len(class_list_features_j)>0:
                    continue
                imgs=torch.stack(class_list_features_j).to(device=self.device).squeeze()
                if len(class_list_features_j)==1:
                    imgs=imgs.unsqueeze(0)
                with torch.no_grad():
                    embs=self.resnet(imgs).cpu()  # 直接将输出移动到 CPU
                totalfeatures[i].append(embs.squeeze())  # 将临时列表赋值给 totalfeatures 的第 i 个元素
                del embs, imgs
                torch.cuda.empty_cache()
        embsss=[]
        for k in range(10):
            tmpdata=[]
            for i in range(len(totalfeatures)):
                tmpdata_i=[]
                for j in range(len(totalfeatures[i])):
                    currembs=totalfeatures[i][j]
                    if currembs.shape[0]>1 and len(currembs.shape)>1:
                        randomchoice=random.randint(0,currembs.shape[0]-1)
                        curremb=currembs[randomchoice,:]
                        tmpdata_i.append(curremb.squeeze())
                    else:
                        tmpdata_i.append(currembs)
                if len(tmpdata_i)<1:
                    continue
                print(len(tmpdata_i),"len_tmp_i")
                tmpdata_i=torch.stack(tmpdata_i).squeeze()
                if len(tmpdata_i.shape)==1:
                    continue
                tmpdata_i1=torch.mean(tmpdata_i,dim=0)
                tmpdata.append(tmpdata_i1)
            embsss.append(tmpdata)
        print(len(embsss),"len_total_modelembs")
        return embsss

    def get_embs(self):
        return self.embs

    def get_query(self,index):
        randomindex=random.randint(0,len(self.embs)-4)
        emb=self.embs[0]#[numgrou,4,1000]
        # emb=self.embs[index%(len(self.embs)-4)]#[numgrou,4,1000]
        label=self.conflabel
        return emb,label

    def get_query_test(self):
        # randomindex=random.randint(0,len(self.embs-3)-1)
        randomindex=len(self.embs)-2
        emb=self.embs[randomindex]#[numgrou,4,1000]
        label=self.conflabel
        return emb,label

    def __getitem__(self, index):
        emb=self.embs[index]   #[numgrou,4,1000]
        print(index,len(emb),"index_len_emb")
        # imgs = self.images[index,:,:,:,:,:]#[numgroup,qnum,4,3,64,64]
        label=self.conflabel
        return emb,label

def get_datasets(datasetpath): #运行太慢，先存到文件夹

    name_num={}
    # num_train=0
    # datasetpath='./../../data/szyhetekm'
    datasets=os.listdir(datasetpath)
    label=0
    featurdict={}

    respath='./../../data/res18.pth'
    device='cuda:0'

    resnet = torchvision.models.resnet18(pretrained=False)
    resnet.load_state_dict(torch.load(respath,map_location='cpu'))
    resnet=resnet.to(device)
    resnet.eval()

    for dataset in datasets:
        print(dataset,"dataset")
        advs=torch.load(os.path.join(datasetpath,dataset))
        itemm=0
        tnum=0
        currembs=[]
        print(advs.keys(),"advs_key")
        nodecenter=advs['node']
        print(len(nodecenter),"len_nodecenter")
        with torch.no_grad():
            for cadv in advs['adv']:
                print(len(cadv),"len_cadv")
                filtered_cadv=[]
                for img in cadv:
                    i=0
                    if isinstance(img, torch.Tensor):
                        filtered_cadv.append(img)
                    else  :
                       filtered_cadv.append(nodecenter[i])
                    # i=i+1
                # filtered_cadv = [img for img in cadv if isinstance(img, torch.Tensor)]
                currimgs=torch.stack(filtered_cadv).to('cuda:0')
                curremb = resnet(currimgs)
                print(curremb.shape,"curr_shape")
                currembs.append(curremb)
            torch.save(currembs,'./../../data/szyhetekmemb2/'+dataset)

    return featurdict


class TrainEmb(Dataset):
    def __init__(self,datasetname,path, label):
        self.device = 'cuda:0'   # Aggregate all embeddings into one list
        self.label=label
        self.embname=datasetname
        self.embs=torch.load(path)
    # def get_query(self):
    #     return self.emb,self.label
    def __len__(self):
        return len(self.embs)

    def __getitem__(self, index):
        emb = self.embs[index]
        label=self.label
        return emb, label


class QueryableConcatDataset(ConcatDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_query(self,index ):
        # Loop through all datasets until we find a sample with the given label
        # for dataset in self.datasets:
        #     for i in range(len(dataset)):
        #         sample, target = dataset[i]
        #         if target == label:
        #             return sample, target
        # Return None if no sample with the given label is found
        samples=[]
        targets=[]
        for dataset in self.datasets:
            sample,target=dataset[:3]
            # sample,target=dataset[index%len(dataset)]
            samples.append(sample)
            targets.append(target)
            # for i in range(len(dataset)):
            #     sample, target = dataset[i]
            #     if target == label:
            #         return sample, target
        return samples,targets
        # return None


# Rest of your code remains the same
def get_datasets_emb():
    datasetpath = './../../data/szyhetekmemb2'
    datasetnames = os.listdir(datasetpath)
    print(datasetnames, "datasetnames")
    item = 0
    all_datasets = []
    for datasetname in datasetnames:
        # if not 'ince' in datasetname:
        #     continue
        path = os.path.join(datasetpath, datasetname)
        dataset = TrainEmb(datasetname, path, item)
        item = item + 1
        all_datasets.append(dataset)
    print(len(all_datasets),"len_al_dataset")
    combined_dataset = QueryableConcatDataset(all_datasets)
    # Create a dataloader from the combined dataset
    batch_size = 64  # Set your desired batch size here
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return dataloader, combined_dataset


if __name__ == '__main__':
    # args=Parser().parse()
    name_num={}
    num_train=0
    datasetpath='./../../data/szyhetekm'
    datasets=os.listdir(datasetpath)
    label=0
    featurdict={}
    get_datasets(datasetpath)