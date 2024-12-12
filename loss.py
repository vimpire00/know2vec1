####################################################################################################
# TANSmodels: Task-Adaptive Neural Network Search with Meta-Contrastive Learning
# Wonyong Jeong, Hayeon Lee, Geon Park, Eunyoung Hyung, Jinheon Baek, Sung Ju Hwang
# github: https://github.com/wyjeong/TANS, email: wyjeong@kaist.ac.kr
####################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosLoss(nn.Module):
    def __init__(self, margin=0.4):
        super(CosLoss, self).__init__()
        self.criterion = nn.CosineEmbeddingLoss(margin=0.4)

    def forward(self, x1,x2,tar):

        loss=self.criterion(x1,x2,tar)

        return torch.sum(loss)


class TripletLossCosine(nn.Module):
    def __init__(self, margin=0.4):
        super(TripletLossCosine, self).__init__()
        self.margin = margin
        self.nmax = 1
        self.contrast = False
        self.device='cuda:0'
        self.criterion = nn.MarginRankingLoss()

    def forward(self, m, q, labels):
        # q = q.squeeze(1)
        scores = torch.mm(m, q.t())  # (44, bathszie)
        diag = scores.diag()  # (160,)
        # scores = (scores - 1 * torch.diag(scores.diag()))
        sorted_model, index_m = torch.sort(scores, 1, descending=True)
        sorted_query, index_q = torch.sort(scores, 0, descending=True)
        # Select the nmax score
        max_q = sorted_query[:self.nmax, :]  # (1, bathsize)
        # print(max_q.size(), "max_q_size")
        # max_m = sorted_model[:, :self.nmax]  # (44, 1)
        tar1 = torch.zeros_like(max_q).to(self.device)
        for i in range(max_q.size(1)):
            tar1[0, i] = 1 if max_q[0, i] == labels[i] else -1
        tar1 = tar1.squeeze()
        neg_q = self.criterion(max_q, diag, tar1)
        loss = neg_q

        return loss




class TripletLossCosine1(nn.Module):
    def __init__(self, margin=0.4):
        super(TripletLossCosine1, self).__init__()
        self.margin = margin
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = nn.MarginRankingLoss(margin=0.4)

    def forward(self, m, q, mlabels, qlabels):
        batchsize = q.size(0)
        num_models = m.size(0)
        # Compute cosine similarity between model and query embeddings
        scores = torch.mm(m, q.t())  # (num_models, batchsize)
        # Rank the scores for each query
        _, sorted_indices = torch.sort(scores, descending=True, dim=1)
        # Calculate the predicted ranks for each query
        # pred_ranks = torch.zeros_like(qlabels, dtype=torch.float).to(self.device)
        # for i in range(batchsize):
        #     pred_ranks[i] = (sorted_indices[i, :] == i).nonzero(as_tuple=True)[0].item() + 1
        # # Compute the target ranks (qlabels should be normalized to start from 1)
        # true_ranks = (qlabels - torch.min(qlabels)) + 1
        # Compute ranking differences
        pred_ranks=sorted_indices
        true_ranks=qlabels
        rank_diff = pred_ranks - true_ranks
        # Create targets for ranking loss
        # We want pred_ranks <= true_ranks, so target should be 1 where diff <= 0 and -1 elsewhere
        target = (rank_diff <= 0).to(torch.float) * 2 - 1
        # Compute loss
        loss = self.criterion(pred_ranks, true_ranks, target)
        
        return loss




# class TripletLossCosine(nn.Module):
#     def __init__(self, margin=0.4):
#         super(TripletLossCosine, self).__init__()
#         self.margin = margin
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#         self.criterion = nn.MarginRankingLoss(margin=0.2)

#     def forward(self, m, q, mlabels, qlabels):
#         batchsize = q.size(0)
#         num_models = m.size(0)
#         # Compute cosine similarity between model and query embeddings
#         scores = torch.mm(m, q.t())  # (num_models, batchsize)
#         # Reshape scores for easier indexing
#         # scores = scores.t()  # (batchsize, num_models)
#         # Calculate positive and negative scores
#         positive_mask = (mlabels.unsqueeze(0) == qlabels.unsqueeze(1))
#         print(positive_mask.shape,"posimask_shape")
#         positive_indices = positive_mask.nonzero(as_tuple=True)
#         print(positive_indices.shape,"shape33")
#         positive_scores = scores[:, positive_indices[1]]
#         # negative_mask = (mlabels.unsqueeze(0) != qlabels.unsqueeze(1))
#         sorted_scores,sorted_indices=torch.sort(scores,descending=True)
#         pre_top=sorted_indices[:,0]
#         target=pre_top.eq(positive_indices[1])
#         pre_scores=scores[:,pre_top]
#         # Original comparison
#         matches = pre_top.eq(positive_indices[1])
#         # Convert boolean tensor to 1s and -1s using sign
#         target = matches.to(torch.float) * 2 - 1
#         target = target.to(self.device)  # Ensure the tensor is on the right device
#         loss = self.criterion(positive_scores,pre_scores, target)

#         return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.8):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, m, q, labels):
    # def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(m,q)
        # distance_negative = F.pairwise_distance(anchor, negative)
        # loss =distance_positive - distance_negative+ self.margin
        loss=distance_positive
        # loss = F.relu(distance_positive - distance_negative + self.margin)
        return torch.sum(loss)


class TripLoss3(nn.Module):
    def __init__(self,args, nmax=1, margin=0.2, contrast=True):
        super(TripLoss3, self).__init__()
        self.margin = margin
        self.nmax = nmax
        self.contrast = True
        self.args=args
        self.criterion = nn.MarginRankingLoss()

    def forward(self, m, q, labels):
        q=q.squeeze(1)
        scores = torch.mm(m, q.t()) # (44, 160)
        diag = scores.diag() # (160,)
        scores = (scores - 1 * torch.diag(scores.diag()))
        sorted_model, index_m = torch.sort(scores, 1, descending=True)
        sorted_query, index_q = torch.sort(scores, 0, descending=True)
        # Select the nmax score
        max_q = sorted_query[:self.nmax, :] # (1, 160)
        max_m = sorted_model[:, :self.nmax] # (160, 1)
        # print(max_q.shape,max_m.shape,"shape22")
        # max_im=index_m[:self.nmax, :]
        # max_iq=index_q[:, :self.nmax]
        # Select the nmax score
        tar1 = torch.zeros_like(max_q).to(self.args.device)
        for i in range(max_q.size(1)):
            tar1[0, i] = 1 if max_q[0, i] == labels[i] else -1
        tar1=tar1.squeeze()
        neg_q=self.criterion(max_q,diag,tar1)
        tar2 = torch.zeros_like(max_q).to(self.args.device)
        neg_m=self.criterion(max_m,diag,tar2)
        if self.contrast:
            loss = neg_m + neg_q
        else:
            loss = neg_m

        return loss


class TripLoss3_HETE(nn.Module):
    def __init__(self, args, margin=0.2):
        super(TripLoss3_HETE, self).__init__()
        self.margin = margin
        self.args = args
        self.criterion = nn.TripletMarginLoss(margin=margin)

    def forward(self, m, q, labels):
        # torch.Size([200, 256]) q_shape torch.Size([200, 26]) labels_shape torch.Size([26, 256]) m_shape
        q = q.squeeze()  # 确保q的维度为[num_queries, feature_dim]
        scores = torch.matmul(m, q.t())  # (num_models, num_queries)[26,200]
        # 获取每个查询的最高分模型的索引
        _, max_indices = torch.max(scores, dim=1)
        # 创建一个布尔掩码，标记出正样本的位置
        positive_mask=labels.to(torch.bool) 
        # 使用掩码从m中选择正样本
        # 为每个查询选择一个正样本（如果有的话）
        # positive_mask=positive_mask.squeeze() #[200,26]
        label_ind=torch.argmax(labels,dim=1)
        # pos_m_indices = torch.where(positive_mask)[1]  # 获取正样本的索引
        pos_m = m[label_ind]
        # 创建一个掩码，排除正样本和当前最高分模型
        # print(max_indices.shape, max_indices[None, :].shape,"max_indices_shape")
        negative_mask = ~positive_mask 
        # 为每个查询选择一个负样本
        # 这里我们选择除了正样本之外得分最高的模型作为负样本
        # 注意：这可能需要根据实际情况进行调整
        neg_m_scores = scores.t() * negative_mask.float()
        neg_m_scores[neg_m_scores == 0] = float('-inf')  # 确保0得分的样本不会被选中
        _, neg_m_indices = torch.max(neg_m_scores, dim=1)
        neg_m = m[neg_m_indices]    # 确保pos_m和neg_m的形状正确，以便与q匹配
        pos_m = pos_m
        neg_m = neg_m
        loss = self.criterion(q, pos_m, neg_m)  # 计算损失
        return loss



class TripLoss3_HETE_Cosine(nn.Module):
    def __init__(self, args, margin=0.2):
        super(TripLoss3_HETE_Cosine, self).__init__()
        self.margin = margin
        self.args = args
        self.criterion = nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, m, q, labels):  # torch.Size([200, 256]) q_shape torch.Size([200, 26]) labels_shape torch.Size([26, 256]) m_shape
        # print(labels.shape,"labels_shape")
        q = q.squeeze()  # 确保q的维度为[num_queries, feature_dim]
        scores = torch.matmul(m, q.t())  # (num_models, num_queries)[26,200]
        _, max_indices = torch.max(scores, dim=1)         # 获取每个查询的最高分模型的索引
        positive_mask=labels.to(torch.bool)  # 创建一个布尔掩码，标记出正样本的位置
        # 使用掩码从m中选择正样本
        # 为每个查询选择一个正样本（如果有的话）
        label_ind=torch.argmax(labels,dim=1)  # 获取正样本的索引  label为one_hot形式
        pos_m = m[label_ind]
        # 创建一个掩码，排除正样本和当前最高分模型
        negative_mask = ~positive_mask
        # 为每个查询选择一个负样本
        # 这里我们选择除了正样本之外得分最高的模型作为负样本
        # 注意：这可能需要根据实际情况进行调整
        neg_m_scores = scores.t() * negative_mask.float()
        neg_m_scores[neg_m_scores == 0] = float('-inf')  # 确保0得分的样本不会被选中
        _, neg_m_indices = torch.max(neg_m_scores, dim=1)
        neg_m = m[neg_m_indices]    # 确保pos_m和neg_m的形状正确，以便与q匹配
        targets = torch.ones_like(q[:, 0]).to(q.device)  # 正样本的目标为1
        # 准备目标tensor用于CosineEmbeddingLoss
        # 将正样本和负样本分别与query比较
        pos_loss = self.criterion(q, pos_m, targets)
        neg_targets = torch.full_like(targets, -1)  # 负样本的目标为-1
        neg_loss = self.criterion(q, neg_m, neg_targets)

        # 计算总损失
        loss = (pos_loss + neg_loss) / 2

        return loss



class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2, contrast=True):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax
        self.contrast = True

    def forward(self, m, q,label, matched=None):
        q=q.squeeze(1)
        scores = torch.mm(m, q.t()) # (160, 160)
        diag = scores.diag() # (160,)
        scores = (scores - 1 * torch.diag(scores.diag()))
        # Sort the score matrix in the query dimension
        # Sort the score matrix in the model dimension
        sorted_model, _ = torch.sort(scores, 1, descending=True)
        sorted_query, _ = torch.sort(scores, 0, descending=True)
        # Select the nmax score
        max_q = sorted_query[:self.nmax, :] # (1, 160)
        max_m = sorted_model[:, :self.nmax] # (160, 1)
        neg_q = torch.sum(torch.clamp(max_q + 
            (self.margin - diag).view(1, -1).expand_as(max_q), min=0))
        neg_m = torch.sum(torch.clamp(max_m + 
            (self.margin - diag).view(-1, 1).expand_as(max_m), min=0))

        if self.contrast:
            loss = neg_m + neg_q
        else:
            loss = neg_m

        return loss