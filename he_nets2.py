import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMCLS(nn.Module):
    def __init__(self,embsize, totclass):
        super(LSTMCLS, self).__init__()
        self.totclass=totclass
        self.classf2=nn.Linear(embsize,self.totclass)
        self.sigmoid=torch.nn.LeakyReLU()
        self.softmax=torch.nn.Softmax(dim=1)
    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=-1, keepdim=True)
        x = torch.div(x, norm2)
        return x
    def forward(self, modelvec):
        out=self.classf2(modelvec)
        classn=self.softmax(out)
        return classn


# class LSTMFE(nn.Module):
#     def __init__(self, input_channels, hidden_size, embsize,num_layers, modelnum):
#         super(LSTMFE, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers * 2
#         self.lstm0 = nn.LSTM(input_channels, self.hidden_size, 2, batch_first=True, bidirectional=True)
#         self.classf1 = nn.Linear(hidden_size * 2,embsize)  # 输出128维向量
#         self.sigmoid = nn.LeakyReLU()
#         self.softmax = nn.Softmax(dim=1)
#         # self.classf2=nn.Linear(embsize,modelnum)

#     def l2norm(self, x):
#         norm2 = torch.norm(x, 2, dim=-1, keepdim=True)
#         x = torch.div(x, norm2)
#         return x

#     def forward(self, data):  # 添加序列长度参数
#         # 对输入序列进行打包，以处理可变长度的序列
#         outs=[]
#         for emb in data:
#             emb=emb.unsqueeze(0)
#             outx,_=self.lstm0(emb)
#             oout=outx[:,-1,:]
#             outs.append(oout.squeeze())

#         outx=torch.stack(outs)
#         outx = self.classf1(outx)
#         # 可选：归一化输出向量
#         out = self.l2norm(outx)
#             # out=self.classf2(outx)
#         return outx ,out


class LSTMFE(nn.Module):
    def __init__(self, input_channels, hidden_size, embsize,num_layers, modelnum):
        super(LSTMFE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers * 2
        self.lstm0 = nn.LSTM(input_channels, self.hidden_size, 2, batch_first=True, bidirectional=True)
        self.lstm1 = nn.LSTM(input_channels, self.hidden_size, 2, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_channels, self.hidden_size, 2, batch_first=True, bidirectional=True)
        self.classf2= nn.Linear(hidden_size * 2,hidden_size)  #avg模式
        # self.classf2= nn.Linear(hidden_size * 6,hidden_size)  concat模式
        self.classf1 = nn.Linear(hidden_size ,embsize)  # 输出128维向量
        self.sigmoid = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=-1, keepdim=True)
        x = torch.div(x, norm2)
        return x

    def forward(self, data):  # 添加序列长度参数
        # 对输入序列进行打包，以处理可变长度的序列
        outs=[]
        for emb in data:
            emb0=emb[0].unsqueeze(0)
            outx,_=self.lstm0(emb0)
            oout0=outx[:,-1,:]
            # outs.append(oout.squeeze())
            emb1=emb[1].unsqueeze(0)
            outx,_=self.lstm1(emb1)
            oout1=outx[:,-1,:]
            emb2=emb[2].unsqueeze(0)
            outx,_=self.lstm2(emb2)
            oout2=outx[:,-1,:]
            # print(oout0.shape,"oout0")
            embs=torch.mean(torch.cat([oout0,oout1,oout2],dim=0),dim=0)
            # print(embs.shape,"embs_shape")
            # embs=torch.cat([oout0,oout1,oout2],dim=1).squeeze()  concat
            outs.append(self.classf2(embs))
        outx=torch.stack(outs)
        outx = self.classf1(outx)
        out = self.l2norm(outx)

        return outx ,out



class QNET(nn.Module):
    def __init__(self, input_channels, hidden_size, embsize,num_layers):
        super(QNET, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers * 2
        self.lstm0 = nn.LSTM(input_channels, self.hidden_size, 2, batch_first=True, bidirectional=True)

        self.classf1 = nn.Linear(hidden_size * 2,embsize)  # 输出128维向量
        self.sigmoid = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=-1, keepdim=True)
        x = torch.div(x, norm2)
        return x

    def forward(self, data,length):  # 添加序列长度参数
        # 对输入序列进行打包，以处理可变长度的序列
        if isinstance(data, list):
            data=torch.stack(data).to('cuda:0')
        packed_output, _ = self.lstm0(data.to('cuda:0'))
        # filtered_data=torch.stack(data).to('cuda:0')
        # packed_output, _ = self.lstm0(filtered_data)
        # 通过线性层将LSTM的输出转换为128维向量
        # output=packed_output[:,length.item(),:]
        # Ensure lengths are on the same device as packed_output
        # lengths = length.to(packed_output.device)
        # Select the last output for each sequence
        length=torch.tensor(length)
        fileter_out=[]
        for i in range(packed_output.shape[0]):
            fileter_out.append(packed_output[i,length[i]-1,:])
        # Create an index tensor for the batch dimension
        # batch_idx = torch.arange(packed_output.size(0), device=lengths.device)
        # Index the last relevant output for each sequence
        # last_outputs = packed_output[:, lengths ,:]
        last_outputs=torch.stack(fileter_out)
        outx = self.classf1(last_outputs)

        return outx