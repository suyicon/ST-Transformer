import torch
import torch.nn as nn
import torch.nn.functional as F

class GATlayer(nn.Module):
    def __init__(self,in_channels,out_channels,dropout,alpha=1e-2,last=False):
        super(GATlayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        #learnable para
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.W = nn.Linear(in_channels,out_channels)

        self.a = nn.Linear(2*out_channels,1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.dropout = dropout
        self.last = last

    def forward(self,
                X,
                adj):
        H = self.W(X)
        N = H.shape[2]
        H_l = H.repeat(1,1,1,N).view(N*N,-1)#overall matrix copy 1 times at row and n times at column-[N,N*out_channels]->[N*N,out_channels],
        H_r = H.repeat(1,1,N,1)#overall matrix copy n times at row and 1 at column
        H_concat = torch.cat(H_l,H_r,dim=-1)
        H_concat =H_concat.view(N,-1,2*self.out_channels)#[N,N,2*out_channels]

        e = self.leakyrelu(self.a(H_concat).squeeze(2))#squeeze is to reduce the dimension if the dimension shape is 1,if not, it won't works [N,N,1]->[N,N]

        zero_vec = -1e12*torch.ones_like(e)#not connected edge be set to infinite negative
        attention = torch.where(adj>0,e,zero_vec)# assign weight to the edge that connected
        attention = F.softmax(attention,dim=1)
        out = torch.matmul(attention,H)

        if self.last:
            return out
        else:
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            return out

        return out

class GAT(nn.Module):
    def __init__(self, C, F, H, dropout):
        super(GAT, self).__init__()
        self.conv1 = GATlayer(C, H, dropout)
        self.conv2 = GATlayer(H, F, dropout)

    def forward(self, X, A):
        hidden = self.conv1(X, A)
        out = self.conv2(hidden, A)
        return out



