import torch
from torch.utils.data import Dataset
import os
import random
import pickle
import logging
from copy import deepcopy
# from learnware_info import BKB_SPECIFIC_RANK_SZY as BKB_SPECIFIC_RANK
from learnware_info import BKB_SPECIFIC_RANK2ID_SZY as  BKB_SPECIFIC_RANK2ID
from learnware_info import TEST_SPECIFIC_RANK
#  DATA_SPECIFIC_RANK
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from load_testacc import  load_ftacc
# UTK_RANK ={'Dogs_inception_v3.pth': 68732.0804497214, 'SUN397_inception_v3.pth': 68992.5814370548, 'Flowers_inception_v3.pth': 69049.72589111328, 'EuroSAT_inception_v3.pth': 69028.93422205308, 'SmallNORB_inception_v3.pth': 69144.05745863914, 'SVHN_inception_v3.pth': 68903.15287511489, 'NABirds_inception_v3.pth': 68994.02895290712, 'Food_inception_v3.pth': 69002.02623555766, 'Cars_inception_v3.pth': 69096.37237189797, 'Caltech101_inception_v3.pth': 68953.46705245972, 'Resisc45_inception_v3.pth': 68958.30126100428, 'CIFAR100_inception_v3.pth': 67801.52064850752, 'CIFAR10_inception_v3.pth': 69183.29327235502, 'AID_inception_v3.pth': 68870.89337517234, 'PACS_inception_v3.pth': 68978.27825209673, 'STL10_inception_v3.pth': 68915.7950571565}

# DS_RANK = {'Dogs_inception_v3.pth': 51.61808431148529, 'SUN397_inception_v3.pth': 51.617711782455444, 'Flowers_inception_v3.pth': 51.61847621202469, 'EuroSAT_inception_v3.pth': 51.61820948123932, 'SmallNORB_inception_v3.pth': 51.619163155555725, 'SVHN_inception_v3.pth': 51.61869078874588, 'NABirds_inception_v3.pth': 51.632581651210785, 'Food_inception_v3.pth': 51.62859708070755, 'Cars_inception_v3.pth': 51.619890332221985, 'Caltech101_inception_v3.pth': 51.61735415458679, 'Resisc45_inception_v3.pth': 51.6200914978981, 'CIFAR100_inception_v3.pth': 51.69091820716858, 'CIFAR10_inception_v3.pth': 51.61154568195343, 'AID_inception_v3.pth': 51.6175240278244, 'PACS_inception_v3.pth': 51.62893086671829, 'STL10_inception_v3.pth': 51.61915719509125}
# DS_RANK ={'Dogs_inception_v3.pth': 0.40997470496222377, 'PACS_resnet50.pth': 0.11098544346168637, 'SUN397_inception_v3.pth': 1.2198598124086857, 'EuroSAT_densenet201.pth': 702.3639440536499, 'Flowers_inception_v3.pth': 0.12072039535269141, 'Resisc45_resnet50.pth': 0.13835706049576402, 'AID_densenet201.pth': 47330.35888671875, 'CIFAR10_resnet50.pth': 0.41517424397170544, 'STL10_densenet201.pth': 92.71806180477142, 'NABirds_resnet50.pth': 0.20065821590833366, 'NABirds_densenet201.pth': 45.27029991149902, 'EuroSAT_inception_v3.pth': 2.865432482212782, 'SmallNORB_inception_v3.pth': 1.4113109093159437, 'Food_densenet201.pth': 6.1135560274124146, 'Food_resnet50.pth': 2.7178894728422165, 'SVHN_inception_v3.pth': 0.16168474103324115, 'PACS_densenet201.pth': 51.889753341674805, 'Resisc45_densenet201.pth': 163.52460980415344, 'Cars_resnet50.pth': 0.1778901496436447, 'Dogs_resnet50.pth': 51.61807984113693, 'NABirds_inception_v3.pth': 0.14424927649088204, 'Food_inception_v3.pth': 0.8103714091703296, 'Flowers_resnet50.pth': 7.44800791144371, 'Cars_inception_v3.pth': 0.2342239604331553, 'SUN397_densenet201.pth': 3067137.98828125, 'Caltech101_inception_v3.pth': 0.7143244380131364, 'Resisc45_inception_v3.pth': 7.605890929698944, 'CIFAR100_inception_v3.pth': 0.6957112811505795, 'Caltech101_resnet50.pth': 0.21703745005652308, 'Caltech101_densenet201.pth': 33535080000.0, 'CIFAR10_inception_v3.pth': 0.2810383099131286, 'CIFAR100_resnet50.pth': 2.6124218478798866, 'EuroSAT_resnet50.pth': 0.4177616792730987, 'AID_inception_v3.pth': 0.21398242097347975, 'SVHN_densenet201.pth': 565360.0830078125, 'Flowers_densenet201.pth': 1001.4615058898926, 'SmallNORB_resnet50.pth': 1.0728382505476475, 'SVHN_resnet50.pth': 0.7302707992494106, 'PACS_inception_v3.pth': 0.39758754428476095, 'Cars_densenet201.pth': 10177533.203125, 'CIFAR100_densenet201.pth': 52.08534598350525, 'Dogs_densenet201.pth': 719.7535991668701, 'SmallNORB_densenet201.pth': 51.85905247926712, 'CIFAR10_densenet201.pth': 2814.682197570801, 'AID_resnet50.pth': 7.171579077839851, 'STL10_resnet50.pth': 0.2444458892568946, 'SUN397_resnet50.pth': 0.8589514996856451, 'STL10_inception_v3.pth': 1.1066026519984007}

# UTK_RANK={'Dogs_inception_v3.pth': 5.577711872609457, 'PACS_resnet50.pth': 5.584234815034413, 'SUN397_inception_v3.pth': 5.072163005719866, 'EuroSAT_densenet201.pth': 5.592222170293899, 'Flowers_inception_v3.pth': 5.538692615327381, 'Resisc45_resnet50.pth': 5.587295479910715, 'AID_densenet201.pth': 5.591094622675578, 'CIFAR10_resnet50.pth': 5.581679340544201, 'STL10_densenet201.pth': 5.565687223307291, 'NABirds_resnet50.pth': 5.1149154389880955, 'NABirds_densenet201.pth': 5.590974075172061, 'EuroSAT_inception_v3.pth': 5.58957373766218, 'SmallNORB_inception_v3.pth': 5.587245456659226, 'Food_densenet201.pth': 5.5918411269415, 'Food_resnet50.pth': 5.57661853831155, 'SVHN_inception_v3.pth': 5.591733169119699, 'PACS_densenet201.pth': 5.590749046107701, 'Resisc45_densenet201.pth': 5.5866048023042225, 'Cars_resnet50.pth': 5.585390930790418, 'Dogs_resnet50.pth': 5.590011907087053, 'NABirds_inception_v3.pth': 5.5995063772178835, 'Food_inception_v3.pth': 5.596299048596337, 'Flowers_resnet50.pth': 5.5825600347609745, 'Cars_inception_v3.pth': 5.580334860084171, 'SUN397_densenet201.pth': 5.588471409970238, 'Caltech101_inception_v3.pth': 5.566330289640881, 'Resisc45_inception_v3.pth': 5.590773493303572, 'CIFAR100_inception_v3.pth': 5.568791501653762, 'Caltech101_resnet50.pth': 5.583475619216192, 'Caltech101_densenet201.pth': 5.583652680896578, 'CIFAR10_inception_v3.pth': 5.596516401599703, 'CIFAR100_resnet50.pth': 5.5912384918772045, 'EuroSAT_resnet50.pth': 5.586101699654545, 'AID_inception_v3.pth': 999999.99, 'SVHN_densenet201.pth': 5.58897105262393, 'Flowers_densenet201.pth': 5.586921949804397, 'SmallNORB_resnet50.pth': 5.576170984395346, 'SVHN_resnet50.pth': 5.585838316301221, 'PACS_inception_v3.pth': 999999.99, 'Cars_densenet201.pth': 5.5824849271138515, 'CIFAR100_densenet201.pth': 5.603483824956985, 'Dogs_densenet201.pth': 5.591217198791675, 'SmallNORB_densenet201.pth': 5.59429814860026, 'CIFAR10_densenet201.pth': 5.589288500685919, 'AID_resnet50.pth': 5.571563919918878, 'STL10_resnet50.pth': 5.592244306446257, 'SUN397_resnet50.pth': 5.556089667038691, 'STL10_inception_v3.pth': 999999.99} 







UTK_RANK={
   'Dogs_inception_v3.pth': 0.05746884, 'PACS_resnet50.pth': 0.0675256, 'SUN397_inception_v3.pth': 0.06917771, 'EuroSAT_densenet201.pth': 0.0748828, 'Flowers_inception_v3.pth': 0.0684475, 'Resisc45_resnet50.pth': 0.0630915, 'AID_densenet201.pth': 0.0619687, 'CIFAR10_resnet50.pth': 0.063332837, 'STL10_densenet201.pth': 0.0634, 'NABirds_resnet50.pth': 0.06339000, 'NABirds_densenet201.pth': 0.0685, 'EuroSAT_inception_v3.pth': 0.07365856, 'SmallNORB_inception_v3.pth': 0.0575233, 'Food_densenet201.pth': 0.0729704, 'Food_resnet50.pth': 0.069158, 'SVHN_inception_v3.pth': 0.067996, 'PACS_densenet201.pth': 0.06345, 'Resisc45_densenet201.pth': 0.0698099, 'Cars_resnet50.pth': 0.0619047, 'Dogs_resnet50.pth': 0.07168, 'NABirds_inception_v3.pth': 0.059647, 'Food_inception_v3.pth': 0.0584035, 'Flowers_resnet50.pth': 0.0719571, 'Cars_inception_v3.pth': 0.066296, 'SUN397_densenet201.pth': 0.05924126, 'Caltech101_inception_v3.pth': 0.0634508, 'Resisc45_inception_v3.pth': 0.060815, 'CIFAR100_inception_v3.pth': 0.07271465, 'Caltech101_resnet50.pth': 0.0753103, 'Caltech101_densenet201.pth': 0.0688426, 'CIFAR10_inception_v3.pth': 0.0617265, 'CIFAR100_resnet50.pth': 0.06410, 'EuroSAT_resnet50.pth': 0.059507, 'AID_inception_v3.pth': 0.064104, 'SVHN_densenet201.pth': 0.06655429, 'Flowers_densenet201.pth': 0.0713702, 'SmallNORB_resnet50.pth': 0.0686039, 'SVHN_resnet50.pth': 0.0618334, 'PACS_inception_v3.pth': 0.076030, 'Cars_densenet201.pth': 0.066736, 'CIFAR100_densenet201.pth': 0.064093, 'Dogs_densenet201.pth': 0.0676144, 'SmallNORB_densenet201.pth': 0.062974, 'CIFAR10_densenet201.pth': 0.0681, 'AID_resnet50.pth': 0.067141, 'STL10_resnet50.pth': 0.0673, 'SUN397_resnet50.pth': 0.0729668, 'STL10_inception_v3.pth': 0.069307
    }
DS_RANK=  {'Dogs_inception_v3.pth': 0.020323001081123948, 'PACS_resnet50.pth': 0.01689290947979316, 'SUN397_inception_v3.pth': 0.018005474703386426, 'EuroSAT_densenet201.pth': 0.02352483570575714, 'Flowers_inception_v3.pth': 0.02140322612831369, 'Resisc45_resnet50.pth': 0.0269766227575019, 'AID_densenet201.pth': 0.023700107703916728, 'CIFAR10_resnet50.pth': 0.012795311340596527, 'STL10_densenet201.pth': 0.020942850096616894, 'NABirds_resnet50.pth': 0.020352010324131697, 'NABirds_densenet201.pth': 0.025883738999255, 'EuroSAT_inception_v3.pth': 0.021661257778760046, 'SmallNORB_inception_v3.pth': 0.022165194968692958, 'Food_densenet201.pth': 0.02811806043609977, 'Food_resnet50.pth': 0.019234634237363935, 'SVHN_inception_v3.pth': 0.026632053777575493, 'PACS_densenet201.pth': 0.02600964216981083, 'Resisc45_densenet201.pth': 0.021328864386305213, 'Cars_resnet50.pth': 0.020494396449066699, 'Dogs_resnet50.pth': 0.020263070473447442, 'NABirds_inception_v3.pth': 0.022589901345781982, 'Food_inception_v3.pth': 0.01755070552462712, 'Flowers_resnet50.pth': 0.017919372476171702, 'Cars_inception_v3.pth': 0.027964674518443644, 'SUN397_densenet201.pth': 0.021052338706795126, 'Caltech101_inception_v3.pth': 0.02626315108500421, 'Resisc45_inception_v3.pth': 0.017496751388534904, 'CIFAR100_inception_v3.pth': 0.019850289390888065, 'Caltech101_resnet50.pth': 0.02136624971171841, 'Caltech101_densenet201.pth': 0.02607545757200569, 'CIFAR10_inception_v3.pth': 0.020631705410778522, 'CIFAR100_resnet50.pth': 0.017237164138350636, 'EuroSAT_resnet50.pth': 0.019461297779344022, 'AID_inception_v3.pth': 0.016314980166498572, 'SVHN_densenet201.pth': 0.02664283092599362, 'Flowers_densenet201.pth': 0.026709967642091215, 'SmallNORB_resnet50.pth': 0.018751699826680124, 'SVHN_resnet50.pth': 0.013930001296103, 'PACS_inception_v3.pth': 0.017660977027844638, 'Cars_densenet201.pth': 0.022326872567646205, 'CIFAR100_densenet201.pth': 0.02249523822683841, 'Dogs_densenet201.pth': 0.023038302606437355, 'SmallNORB_densenet201.pth': 0.019162656099069864, 'CIFAR10_densenet201.pth': 0.01863758807303384, 'AID_resnet50.pth': 0.016584896366111934, 'STL10_resnet50.pth': 0.026114267529919744, 'SUN397_resnet50.pth': 0.02139620337402448, 'STL10_inception_v3.pth': 0.01707602641545236} 


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
            
def get_rankings(values):   
# 创建一个列表，初始化为0，长度与原始列表相同
    rankings = [0] * len(values)
    # 对列表中的值及其原始索引进行排序
    sorted_pairs = sorted(enumerate(values), key=lambda x: x[1])
    # 初始化排名
    rank = 0
    # 上一个值，用于检测重复值
    prev_value = None
    # 遍历排序后的值和索引
    for i, (_, value) in enumerate(sorted_pairs):
        if value != prev_value:
            # 如果当前值不同于前一个值，增加排名
            rank = i + 1
        rankings[sorted_pairs[i][0]] = rank  # 设置原始位置的排名
        prev_value = value

    return rankings

class LearnwareDataset(Dataset):
    __heterogeneous_sampled_minnum__ = None
    __heterogeneous_sampled_maxnum__ = None
    __heterogeneous_sampled_fixnum__ = None
    __heterogeneous_prefetch_rank__ = None

    def __init__(self, args, stype, continuous_label=False, samples=None, heterogeneous=False):
        super().__init__()
        LearnwareDataset.__heterogeneous_sampled_minnum__ = args.heterogeneous_sampled_minnum
        LearnwareDataset.__heterogeneous_sampled_maxnum__ = args.heterogeneous_sampled_maxnum

        self.args=args
        modelnameso = os.listdir('./../../data/szyhetekmemb2')
        global inception_indices
        inception_indices = [i for i, name in enumerate(modelnameso)]
        # inception_indices = [i for i, name in enumerate(modelnameso) if 'inception' in name]
        self.BKB_SPECIFIC_RANK=[]
        for modelname in modelnameso:
            # if 'CIFAR10' in modelname and not 'CIFAR100' in modelname:
            #         continue
            # if 'SmallNORB_inception_v3.pth' in modelname:
            #     continue
            # if not 'ince' in modelname:
            #     continue
            self.BKB_SPECIFIC_RANK.append(modelname[:-5])
        self.BKB_SPECIFIC_RANK2ID_SZY={i: j for i, j in zip(self.BKB_SPECIFIC_RANK, range(len(self.BKB_SPECIFIC_RANK)))}
        global BKB_SPECIFIC_RANK2ID
        BKB_SPECIFIC_RANK2ID=self.BKB_SPECIFIC_RANK2ID_SZY
        self.continuous_label = continuous_label
        self.prototype_maxnum = 100
        self.heterogeneous = heterogeneous
        self.stype = stype
        if self.stype=='train':
            self.DATA_SPECIFIC_RANK,self.samples=self.get_data_sp_rank()
            # print(self.DATA_SPECIFIC_RANK)
        elif self.stype=='test':
            self.DATA_SPECIFIC_RANK,self.samples,self.TESTACC=self.get_test_datarank()
        else:
            print("test error")

    def get_test_datarank(self,):
        samples=[]
        cur_datasets_path = self.args.test_dataset_path
        # cur_datasets=os.listdir(cur_datasets_path)   
        cur_datasets=['UTKFace','dSprites']
        DATA_SPECIFIC_RANK={}
        # DDATA_SPECIFIC_RANK = torch.load('./../../data/szyhetetest4modeltestrank.npy', map_location='cpu')
        # for key in  DDATA_SPECIFIC_RANK.keys():
        #     DATA_SPECIFIC_RANK[key]=DDATA_SPECIFIC_RANK[key]   
                
        fixed_gt_samples_num = 0
        for i_dataset in cur_datasets:
            cur_samples, cur_fixed_gt_samples = [], []
            cur_path = os.path.join(self.args.test_dataset_path, i_dataset)
            cur_ready_path = (cur_path, i_dataset)
            cur_fixed_gt_samples.append(cur_ready_path)
            cur_samples.append(cur_ready_path)

            if self.stype == 'train' and self.args.fixed_gt_size_threshold != 0:
                cur_fixed_gt_samples = random.sample(cur_fixed_gt_samples, min(len(cur_fixed_gt_samples), self.args.fixed_gt_size_threshold))
                fixed_gt_samples_num += len(cur_fixed_gt_samples)
            if len(cur_fixed_gt_samples) != 0:
                cur_samples += cur_fixed_gt_samples
                random.shuffle(cur_samples)
            if (self.stype == 'test' or self.stype == 'val') and self.args.test_size_threshold != 0:
                test_copy_samples = deepcopy(cur_samples)
                random.shuffle(test_copy_samples)
                cur_samples = test_copy_samples[: self.args.test_size_threshold]
            samples.extend(cur_samples)
        print(len(samples),"len_testsamples")
        FT_ACC={}

        for taskname in ['Aircraft','DTD','Pet']:
            FT_ACC[taskname]=load_ftacc(self.BKB_SPECIFIC_RANK2ID_SZY,taskname)
            #由于训练集中更改了排名的计算方式，更改一下测试集中排名的计算方式
                # ft_acc=load_ftacc(self.BKB_SPECIFIC_RANK2ID_SZY,taskname)
                # # sortacc,isortacc=torch.sort(torch.stack(ft_acc))
                # # FT_ACC[taskname]=isortacc.tolist()
                # ft_acc_rankings = get_rankings(ft_acc)
                # FT_ACC[taskname]=ft_acc_rankings
        return DATA_SPECIFIC_RANK,samples,FT_ACC


    def get_data_sp_rank(self,):
        samples=[]
        cur_datasets_path = self.args.train_dataset_path
        cur_datasets = os.listdir(cur_datasets_path)
        # Assuming self.DATA_SPECIFIC_RANK is loaded as mentioned before
        # DDATA_SPECIFIC_RANK = torch.load('./../../data/szyhetetest4modeltrainrank.npy', map_location='cpu')
        DDATA_SPECIFIC_RANK = torch.load('./../../data/szyhetemlprealtrainrank.npy', map_location='cpu')
        # Adjust indices in self.DATA_SPECIFIC_RANK based on removals
        DATA_SPECIFIC_RANK={}   
        # 找到排名中的最大值   
        # 遍历排名字典，转换每个排名为0到1之间的数值
        # for key in list(DDATA_SPECIFIC_RANK.keys()):
        #     # max_rank = max(DDATA_SPECIFIC_RANK[key].values())
        #     # 转换公式：(最大排名 - 当前排名 + 1) / (最大排名 + 1)
        #     max_rank=len(DDATA_SPECIFIC_RANK[key])
        #     crank=DDATA_SPECIFIC_RANK[key]
        #     rerank=[]
        #     for rank in crank:  
        #         normalized_rank = (max_rank - rank + 1) / (max_rank + 1)
        #         rerank.append(normalized_rank)
        #     DATA_SPECIFIC_RANK[key]=torch.stack(rerank)
        # # 输出转换后的字典
        # print(DATA_SPECIFIC_RANK)
        for key in  DDATA_SPECIFIC_RANK.keys():
            if not 'tt' in key:
                continue
            # print(DDATA_SPECIFIC_RANK[key].shape,"ddshape")
            tensortt= torch.stack([DDATA_SPECIFIC_RANK[key][i] for i in inception_indices])
            DATA_SPECIFIC_RANK[key]=tensortt

        fixed_gt_samples_num = 0
        for i_dataset in cur_datasets:
            cur_samples, cur_fixed_gt_samples = [], []
            if self.stype == 'test':
                cur_path = os.path.join(self.args.test_dataset_path, i_dataset)
            else:
                if not i_dataset+'.pth' in DATA_SPECIFIC_RANK.keys():
                    continue
                cur_path = os.path.join(self.args.data_url, i_dataset)
            cur_ready_path = (cur_path, i_dataset)
            cur_fixed_gt_samples.append(cur_ready_path)
            cur_samples.append(cur_ready_path)

            if self.stype == 'train' and self.args.fixed_gt_size_threshold != 0:
                cur_fixed_gt_samples = random.sample(cur_fixed_gt_samples, min(len(cur_fixed_gt_samples), self.args.fixed_gt_size_threshold))
                fixed_gt_samples_num += len(cur_fixed_gt_samples)
            if len(cur_fixed_gt_samples) != 0:
                cur_samples += cur_fixed_gt_samples
                random.shuffle(cur_samples)
            if (self.stype == 'test' or self.stype == 'val') and self.args.test_size_threshold != 0:
                test_copy_samples = deepcopy(cur_samples)
                random.shuffle(test_copy_samples)
                cur_samples = test_copy_samples[: self.args.test_size_threshold]

            samples.extend(cur_samples)

        if self.stype == 'train':
            print(f'Train fixed samples: {fixed_gt_samples_num}')

        return DATA_SPECIFIC_RANK,samples

    def pad(self,ffea):
        # Assuming ffea is a list of tensors with potentially varying lengths
        # Each tensor in ffea is of shape [n, 1000], where n can vary
        # First, determine the number of elements in ffea
        num_elements = len(ffea)
        # If the number of elements is less than 50, pad the sequence
        if num_elements < 100:
            # Create a list of tensors to pad
            tensors_to_pad = ffea
            # Calculate the number of additional tensors needed
            num_additional_tensors = 100 - num_elements
            # Create additional zero tensors to append
            additional_tensors = [torch.zeros(1000)] * num_additional_tensors
            # Concatenate the original tensors with the additional zero tensors
            padded_ffea = tensors_to_pad + additional_tensors
            # Convert the list back to a tensor using pad_sequence
            ffea = pad_sequence(padded_ffea, batch_first=True)
            padlength=num_additional_tensors
        # If the number of elements is greater than 50, truncate the sequence
        elif num_elements > 100:
            # Simply slice the first 50 tensors
            ffea = ffea[:100]
            padlength=0
            ffea=torch.stack(ffea)
        else:
            padlength=0
        # If the number of elements is exactly 50, no action is required
        # Now, convert the list of tensors to a stacked tensor if necessary
        # This step might be redundant if you've already used pad_sequence
            ffea = torch.stack(ffea)
        # Ensure the final shape is [50, 1000]
        # assert ffea.shape == (50, 1000), "ffea shape is not [50, 1000]"
        return  ffea,padlength

    def pad_test(self,ffea):
        # Assuming ffea is a list of tensors with potentially varying lengths
        # Each tensor in ffea is of shape [n, 1000], where n can vary
        # First, determine the number of elements in ffea
        num_elements = len(ffea)
        # If the number of elements is less than 50, pad the sequence
        if num_elements < 100:
            # Create a list of tensors to pad
            tensors_to_pad = ffea
            # Calculate the number of additional tensors needed
            num_additional_tensors = 100 - num_elements
            # Create additional zero tensors to append
            additional_tensors = [torch.zeros(1000)] * num_additional_tensors
            # Concatenate the original tensors with the additional zero tensors
            padded_ffea = tensors_to_pad + additional_tensors
            # Convert the list back to a tensor using pad_sequence
            ffea = pad_sequence(padded_ffea, batch_first=True)
            padlength=num_additional_tensors
        # If the number of elements is greater than 50, truncate the sequence
        # elif num_elements > 50:
        #     # Simply slice the first 50 tensors
        #     ffea = ffea[:50]
        #     padlength=0
        #     ffea=torch.stack(ffea)
        else:
            padlength=0
            ffea=torch.stack(ffea)
        # If the number of elements is exactly 50, no action is required
        # Now, convert the list of tensors to a stacked tensor if necessary
        # This step might be redundant if you've already used pad_sequence
            # ffea = torch.stack(ffea)
        # Ensure the final shape is [50, 1000]
        # assert ffea.shape == (50, 1000), "ffea shape is not [50, 1000]"
        return  ffea,padlength

    def __getitem__(self, index):
        """
        return: [num_prototypes, dim], [num_learnware]
        """
        cur_discrete_type = 'FTRank'

        if self.stype=='train' or  self.stype=='val' :
            xx = torch.load(self.samples[index][0],map_location='cpu') #x[0]:数据特征[85,3072],85是类别数
            tlabel=xx['Dogs_inception_v3.pth']['tar']
            feas=xx['res18']
            unique_labels = np.unique(tlabel)  # 对于每个类别，计算该类别的特征平均值
            class_means = {}
            ffea=[]
            for label in unique_labels:
                indices = torch.where(tlabel == label)[0]   # 获取属于当前类别的样本索引
                class_features = feas[indices]  # 根据索引选择特征向量
                mean_features = torch.mean(class_features, dim=0)   # 计算平均特征向量
                class_means[label.item()] = mean_features # 存储平均特征向量
                ffea.append(mean_features)
            ffea,length=self.pad(ffea)
            query_key = self.samples[index][1] + '.pth'
            try:
                label = self.DATA_SPECIFIC_RANK[query_key]
            except KeyError:
                print(f"Key not found in DATA_SPECIFIC_RANK: {query_key}")
                label=0
            return ffea, label,100-length 

        else:
            query_key=self.samples[index][1]
            xx=torch.load(self.samples[index][0]+'.pth',map_location='cpu')
            if not isinstance(xx, list):
                tmpfeatures=[]
                for k,va in zip(xx.keys(),xx.values()):
                    currvalues=torch.stack(va)
                    ccv=torch.mean(currvalues,dim=0)
                    tmpfeatures.append(ccv)
                label=self.TESTACC[query_key]
                ffea,length=self.pad_test(tmpfeatures)
            else:
                ffea=torch.cat(xx,dim=0)
                index_item={}
                if 'UTKFace' in self.samples[index][0]:
                    SELECTED_ACCS=[]
                    for k in self.BKB_SPECIFIC_RANK:
                        if not k in UTK_RANK.keys():
                            continue
                            # SELECTED_ACCS.append(torch.tensor(0))
                        else:
                            SELECTED_ACCS.append(torch.tensor(UTK_RANK[k]))
                else:
                    SELECTED_ACCS=[]
                    for k in self.BKB_SPECIFIC_RANK:
                        if not k in  DS_RANK.keys():
                            continue
                            # SELECTED_ACCS.append(torch.tensor(0))
                        else:
                            SELECTED_ACCS.append(torch.tensor(DS_RANK[k]))
                        # SELECTED_ACCS.append(torch.tensor(DS_RANK[k]) )
                label=SELECTED_ACCS
                length=0
            return ffea,label,100-length

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        print("collate_fn")
        x_uni_hete, cur_rank,pad_length = zip(*batch)
        # ret_x=torch.stack(x_uni_hete)
        print("collated_ftn")
        first_elements = [sublist[0].cpu() for sublist in cur_rank]
        # cpu_xuni=[]
        qlength=[]
        for ele in x_uni_hete:
            ele.cpu()
            qlength.append(100-pad_length)
        print(len(x_uni_hete),len(first_elements),"lenn")
        return x_uni_hete, torch.stack(first_elements).cpu(),qlength