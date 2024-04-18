"""
    Implementation of SubModel
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv


class ASPPConv(nn.Sequential):
    """ ASPP Convolution """
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    """ ASPP Pooling """
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """ ASPP """
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        ]

        rates = tuple(atrous_rates)

        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):
    """ DeepLab Head """
    def __init__(self, in_channels):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36])
        )


class DeeplabNeck(nn.Module):
    """ Deeplab Neck """
    def __init__(self, out_channels=25):
        super(DeeplabNeck, self).__init__()
        self.conv = nn.Conv2d(256, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(5)
            
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CommonClassifierHead(nn.Sequential):
    """ Classifier Head """
    def __init__(self, in_channels=625, out_channels=64):
        super(CommonClassifierHead, self).__init__(
            nn.Linear(in_channels, out_channels, bias=True)
        )


class ClassifierHeadAction(nn.Sequential):
    """ Classifier Head """
    def __init__(self, in_channels=64, out_channels=4):
        super(ClassifierHeadAction, self).__init__(
            nn.Linear(in_channels, out_channels, bias=True)
        )


class ClassifierHeadReason(nn.Sequential):
    """ Classifier Head """
    def __init__(self, in_channels=64, out_channels=21):
        super(ClassifierHeadReason, self).__init__(
            nn.Linear(in_channels, out_channels, bias=True)
        )


class CrossSemantic(nn.Module):
    """ Cross Label Semantic and Image Semantic """
    def __init__(self, num_classes=25, txt_feature_dim=768, intermediary_dim=256):
        super(CrossSemantic, self).__init__()
        self.num_classes = num_classes
        self.txt_feature_dim = txt_feature_dim
        self.intermediary_dim = intermediary_dim

        self.img_embedding = nn.Linear(self.num_classes, self.intermediary_dim, bias=False)
        self.txt_embedding = nn.Linear(self.txt_feature_dim, self.intermediary_dim, bias=False)

        self.fc = nn.Linear(self.intermediary_dim, self.intermediary_dim)
        self.attention = nn.Linear(self.intermediary_dim, 1)
    
    def forward(self, img_feature_map, txt_features):
        """
            args:
                img_feature_map: [Batch_size, channel, height, width]
                txt_features: [num_class, 768]
        """
        batch_size = img_feature_map.shape[0]
        height, width = img_feature_map.shape[2], img_feature_map.shape[3]
        img_feature_map = img_feature_map.permute(0, 2, 3, 1)                               # [Batch_size, height, width, channel]
        img_feature_map = img_feature_map.unsqueeze(3)                                      # [Batch_size, height, width, 1, channel]

        # img embedding
        f_wh_feature = self.img_embedding(img_feature_map)                                  # [Batch_size, height, width, 1, intermediary_dim]

        # txt embedding
        f_wd_feature = self.txt_embedding(txt_features)                                     # [num_class, intermediary_dim]
        
        # attention
        lb_feature = torch.tanh(f_wh_feature * f_wd_feature)                                # [Batch_size, height, width, num_classes, intermediary_dim]
        lb_feature = self.fc(lb_feature)                                                    # [Batch_size, height, width, num_classes, intermediary_dim]
        coefficient = self.attention(lb_feature)                                            # [Batch_size, height, width, num_classes, 1]
        coefficient = coefficient.view(batch_size, -1, self.num_classes)                    # [Batch_size, height * width, num_classes, 1]
        coefficient = F.softmax(coefficient, dim=1)                                         # [Batch_size, height * width, num_classes, 1]
        coefficient = coefficient.view(batch_size, height, width, self.num_classes, 1)      # [Batch_size, height, width, num_classes, 1]

        # output
        attention_output = img_feature_map * coefficient                                    # [Batch_size, height, width, num_classes, channel]
        attention_output = torch.sum(torch.sum(attention_output, 1) ,1)                     # [Batch_size, num_classes, channel]
        return attention_output, coefficient
    

class GNN(nn.Module):
    """ Graph Neural Network """
    def __init__(self, input_dim=256, hidden_dim=32, output_dim=16, heads_mum=8):
        super(GNN, self).__init__()
        self.layer1 = GATConv(in_channels=input_dim, out_channels=hidden_dim, heads=heads_mum)
        self.layer2 = GATConv(in_channels=hidden_dim*heads_mum, out_channels=output_dim, heads=heads_mum)
    
    def forward(self, x, edge_index, edge_attr):
        x = self.layer1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        # x = F.dropout(x, p=0.3, training=self.training)
        x, (_, alpha) = self.layer2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        return x, alpha


def getModelSize(model):
    """ Get Model Size """
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print(f'Model Size is: {all_size:.3f}MB')


if __name__ == "__main__":
    a = torch.rand(2, 256, 90, 160)
    b = torch.rand(25, 768)
    attention_moduld = CrossSemantic(num_classes=25)
    gnn_moduld = GNN()
    getModelSize(attention_moduld)  # 1.02M
    getModelSize(gnn_moduld)
    print(0)
