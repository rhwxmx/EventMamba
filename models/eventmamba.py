import torch
import torch.nn as nn
from mamba_layer import MambaBlock
from modules import LocalGrouper
import torch.nn.functional as F

##### Define the attention mechanism #####
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, output):
        attn_weights = self.linear(output).squeeze(-1)
        attn_probs = torch.softmax(attn_weights, dim=1)
        return attn_probs
##### Define the linear layers, just utilized to embdedding the coordinate#####    
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
##### Define the residual layers, which abstract the events features#####    
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
    def __init__(self,num_classes):
        super().__init__()
        # self.feature_list = [32,64,128,256]
        ##### define the feature list, the dimension of the feature list in different stages#####
        # self.feature_list = [16,32,64,128]
        self.feature_list = [24,48,96,192]
        ##### define the centroid list, the number of the centroid list in different stages#####
        self.group_number = [512,256,128,64]
        ##### define the neighbors list, the number of the neighbors list in different stages#####
        self.neighbors = [24,24,24,24]
        self.stages = 3
        self.bimamba_type = 'v2'
        self.local_grouper_list = nn.ModuleList()
        self.local_conv_list = nn.ModuleList()
        self.aggregation_list = nn.ModuleList()
        self.mamba_list = nn.ModuleList()
        self.global_conv_list = nn.ModuleList()
        ##### define the embedding layer#####
        self.embed_dim = Linear1Layer(3,self.feature_list[0],1)
        self.attention = Attention(self.feature_list[-1])
        for i in range(self.stages):
            local_grouper = LocalGrouper(self.feature_list[i], self.group_number[i], self.neighbors[i], use_xyz=False, normalize="anchor")
            self.local_grouper_list.append(local_grouper)
            local_conv = Linear2Layer(self.feature_list[i+1],1,1)
            self.local_conv_list.append(local_conv)
            aggregation = Attention(self.feature_list[i+1])
            self.aggregation_list.append(aggregation)
            mamba = MambaBlock(dim = self.feature_list[i+1], layer_idx = 0, bimamba_type = self.bimamba_type)
            self.mamba_list.append(mamba)
            global_conv = Linear2Layer(self.feature_list[i+1],1,1)
            self.global_conv_list.append(global_conv)

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_list[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x: torch.Tensor):
        ##### the shape of x is [B, 3, N] #####
        ##### the shape of xyz is [B, N, 3] #####
        xyz = x.permute(0,2,1)
        batch_size, _, _ = x.size()
        x = self.embed_dim(x)
        x = x.permute(0,2,1)
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x)
            x = x.permute(0, 1, 3, 2)
            b, n, d, s = x.size()
            x = x.reshape(-1,d,s)
            x = self.local_conv_list[i](x)
            x = x.permute(0,2,1)
            att = self.aggregation_list[i](x)
            x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
            x = x.reshape(b, n, -1)
            x,_= self.mamba_list[i](x)
            x = x.permute(0, 2, 1)
            x = self.global_conv_list[i](x)
            x = x.permute(0,2,1)
        attn = self.attention(x)
        x = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        x = self.classifier(x)     
        return x   

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    model =  EventMamba(num_classes=10).cuda()
    model.eval()
    flops = FlopCountAnalysis(model, torch.rand(1, 3, 1024).cuda())
    print(flop_count_table(flops, max_depth=1))