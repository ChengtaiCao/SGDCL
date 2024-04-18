""" 
    Implementation of Model 
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from fightingcv_attention.attention.CBAM import CBAMBlock

from model.Backbone import resnet50
from model.SubModel import DeepLabHead, DeeplabNeck, CrossSemantic, GNN, CommonClassifierHead, ClassifierHeadAction, ClassifierHeadReason


class IntermediateLayerGetter(nn.ModuleDict):
    """ obtain intermediate layer """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers
    
    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FinalModel(nn.Module):
    """ Final whole model """
    def __init__(self, backbone, cbam, assp, neck, attention_module, gnn, common_classifier_head, classifier_head_action, classifier_head_reason):
        super(FinalModel, self).__init__()
        """
            backbone: 89.879M
            cbam: 2.000M
            classifier: 59.273M
            neck: 0.220M
            attention_module: 1.026M
            gnn: 0.781
            common_classifier_head: 0.031M
            classifier_head_action: 0.001M
            classifier_head_reason: 0.005M
        """
        self.backbone = backbone
        self.cbam = cbam 
        self.classifier = assp
        self.neck = neck
        self.attention_module = attention_module
        self.gnn = gnn
        self.common_classifier_head = common_classifier_head
        self.classifier_head_action = classifier_head_action
        self.classifier_head_reason = classifier_head_reason
    
    def forward(self, x, label_embedding, adj_COO, edge_attr):
        result = OrderedDict()

        # backbone
        features = self.backbone(x)
        x = features["out"]                             # [Batch_size, 2048, 90, 160]

        # cbam
        x = self.cbam(x)                                # [Batch_size, 2048, 90, 160]

        # classifier
        x = self.classifier(x)                          # [Batch_size, 256, 90, 160]

        # neck
        x = self.neck(x)                                # [Batch_size, 25, 18, 32]

        # attention_module    
        x, coefficient = self.attention_module(x, label_embedding)   # [Batch_size, 25, 25]

        # gnn
        batch_size = x.shape[0]
        num_classes = x.shape[1]
        x = x.view(batch_size * num_classes, -1)        # [Batch_size * 25, 25]
        x, alpha = self.gnn(x, adj_COO, edge_attr)      # [Batch_size * 25, 128]
        alpha = torch.mean(alpha, dim=1)                # [Batch_size * 25, Num_Edges (+ self-loop)]
        x = x.view(batch_size, num_classes, -1)         # [Batch_size, 25, output_dim * heads_mum]
        x = torch.flatten(x, start_dim=1)               # [Batch_size, 25 * output_dim * heads_mum]

        # common_classifier_head
        x = self.common_classifier_head(x)              # [Batch_size, 64]

        # classifier_head_action
        y1 = self.classifier_head_action(x)             # [Batch_size, 4]

        # classifier_head_reason
        y2 = self.classifier_head_reason(x)             # [Batch_size, 21]

        result["out"] = [y1, y2, alpha, coefficient]
        return result["out"]


def get_model(config, pretrained=True):
    """ Get Whole Model """
    out_inplanes = config["out_inplanes"]

    action_class = config["action_class"]
    reason_class = config["reason_class"]
    total_class = action_class + reason_class

    attention_dim = config["attention_dim"]
    classifier_hidden_dim = config["classifier_hidden_dim"]

    gnn_hidden_dim = config["GNN"]["hidden_dim"]
    gnn_output_dim = config["GNN"]["output_dim"]
    gnn_head_num = config["GNN"]["heads_mum"]

    # Backbone
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])
    return_layers = {'layer4': 'out'}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # CBAM
    cbam = CBAMBlock(channel=out_inplanes, reduction=16, kernel_size=7)

    # ASPP
    assp = DeepLabHead(out_inplanes)

    # Head
    neck = DeeplabNeck(out_channels=total_class)

    # CrossSemantic
    attention_module = CrossSemantic(num_classes=total_class,
                                     txt_feature_dim=768,
                                     intermediary_dim=attention_dim)
    
    # GNN
    gnn = GNN(input_dim=total_class, 
              hidden_dim=gnn_hidden_dim, 
              output_dim=gnn_output_dim, 
              heads_mum=gnn_head_num)

    # Common Classifier Head
    common_classifier_head = CommonClassifierHead(gnn_output_dim*gnn_head_num*total_class, 
                                                  classifier_hidden_dim)

    # Seperate Classifier Heads
    classifier_head_action = ClassifierHeadAction(in_channels=classifier_hidden_dim,
                                                  out_channels=action_class)
    classifier_head_reason = ClassifierHeadReason(in_channels=classifier_hidden_dim,
                                                  out_channels=reason_class)

    model = FinalModel(backbone, cbam, assp, neck, attention_module, gnn, common_classifier_head, classifier_head_action, classifier_head_reason)

    if pretrained:
        weights_dict = torch.load("./weight/bdd10k_resnet50_1.pth", map_location='cpu')
        weights_dict = weights_dict["model"]
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys1: ", missing_keys)
            print("unexpected_keys1: ", unexpected_keys)

    return model


if __name__ == "__main__":
    net = get_model()
    input_tensor = torch.FloatTensor(2, 3, 720, 1280)
    output = net(input_tensor)
    print(output)
