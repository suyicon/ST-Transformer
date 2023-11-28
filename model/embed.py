from torch import nn


class Embedding(nn.Module):
    def __init__(self, d_feature, d_model):
        super(Embedding, self).__init__()
        self.conv = nn.Conv2d(d_feature, d_model, (1, 1))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        #print(x.shape)torch.Size([32, 1, 12, 25])
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        #print(x.shape)torch.Size([32, 12, 25, 64])
        return x
