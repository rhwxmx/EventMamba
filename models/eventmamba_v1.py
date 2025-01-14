import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from modules import LocalGrouper
from mamba_layer import MambaBlock

##### Define the attention mechanism #####
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, output):
        attn_weights = self.linear(output).squeeze(-1)
        attn_probs = torch.softmax(attn_weights, dim=1)
        return attn_probs
    
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=int(in_channels/2),
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(in_channels/2)),
            self.act
        )
        self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(in_channels/2), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class EventMamba(nn.Module):
    def __init__(self,num_classes=6,num=1024):
        super().__init__()
        self.n = num
        bimamba_type = "v2"
        # bimamba_type = None
        # self.feature_list = [6,16,32,64]
        # self.feature_list = [6,32,64,128]
        self.feature_list = [6,64,128,256]
        # self.feature_list = [6,128,256,512]
        self.group = LocalGrouper(3, 512, 24, False, "anchor")
        self.group_1 =LocalGrouper(self.feature_list[1], 256, 24, False, "anchor")
        self.group_2 =LocalGrouper(self.feature_list[2], 128, 24, False, "anchor")
        # self.group = LocalGrouper(3, 1024, 24, False, "anchor")
        # self.group_1 =LocalGrouper(self.feature_list[1], 512, 24, False, "anchor")
        # self.group_2 =LocalGrouper(self.feature_list[2], 256, 24, False, "anchor")
        self.embed_dim = Linear1Layer(self.feature_list[0],self.feature_list[1],1)
        self.conv1 = Linear2Layer(self.feature_list[1],1,1)
        self.conv1_1 = Linear2Layer(self.feature_list[1],1,1)
        self.conv2 = Linear2Layer(self.feature_list[2],1,1)
        self.conv2_1 = Linear2Layer(self.feature_list[2],1,1)
        self.conv3 = Linear2Layer(self.feature_list[3],1,1)
        self.conv3_1 = Linear2Layer(self.feature_list[3],1,1)
        self.mamba1 = MambaBlock(dim = self.feature_list[1], layer_idx = 0, bimamba_type = bimamba_type)
        self.mamba2 = MambaBlock(dim = self.feature_list[2], layer_idx = 1,bimamba_type = bimamba_type)
        self.mamba3 = MambaBlock(dim = self.feature_list[3], layer_idx = 2,bimamba_type = bimamba_type)
        self.attention_1 = Attention(self.feature_list[1])
        self.attention_2 = Attention(self.feature_list[2])
        self.attention_3 = Attention(self.feature_list[3])
        self.attention_4 = Attention(self.feature_list[3])
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_list[3], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            # nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        xyz = x.permute(0,2,1)
        batch_size, _, _ = x.size()
        xyz, x = self.group(xyz, x.permute(0, 2, 1))
        x= x.permute(0, 1, 3, 2)
        b, n, d, s = x.size()
        x = x.reshape(-1,d,s)
        x = self.embed_dim(x)
        x = self.conv1(x)
       
        x = x.permute(0,2,1)
        att = self.attention_1(x)
        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
        x = x.reshape(b, n, -1)
        x , _= self.mamba1(x)
        x = x.permute(0,2,1)
        x = self.conv1_1(x)
        x = x.permute(0,2,1)
       
        xyz,x = self.group_1(xyz, x)
        x= x.permute(0, 1, 3, 2)
        b, n, d, s = x.size()
        x = x.reshape(-1,d,s)
        x = self.conv2(x)
        x = x.permute(0,2,1)
        att = self.attention_2(x)
        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
        x = x.reshape(b, n, -1)
        x , _= self.mamba2(x)
        x = x.permute(0,2,1)
        x = self.conv2_1(x)
        x = x.permute(0,2,1)

        xyz,x = self.group_2(xyz, x)
        x= x.permute(0, 1, 3, 2)
        b, n, d, s = x.size()
        x = x.reshape(-1,d,s)
        x = self.conv3(x)
        x = x.permute(0,2,1)
        att = self.attention_3(x)
        x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
        x = x.reshape(b, n, -1)
        x,_= self.mamba3(x)
        x = x.permute(0,2,1)
        x = self.conv3_1(x)
        x = x.permute(0,2,1)

        attn = self.attention_4(x)
        x = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        x = self.classifier(x)

        # x = torch.max(x, 2, keepdim=True)[0]
        # x = x.view(-1, self.feature_list[-1])
        # x = self.classifier(x)
        return x