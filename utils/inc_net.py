import copy
import logging
import torch
from torch import nn
from backbone.linears import CosineLinear, EaseCosineLinear, SimpleLinear

import timm

def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    
    if name in ("vit_base_patch16_224_multi_adapter"):
        ffn_num = args["ffn_num"]
        if args["model_name"] == "ours":
            from backbone import vit_multi_adapter
            from easydict import EasyDict
            tuning_config = EasyDict(
                # AdaptFormer
                ffn_adapt=True,
                ffn_option="parallel",
                ffn_adapter_layernorm_option="none",
                ffn_adapter_init_option="lora",
                ffn_adapter_scalar="0.1",
                ffn_num=ffn_num,
                d_model=768,
                # VPT related
                vpt_on=False,
                vpt_num=0,
                _device = args["device"][0]
            )
            if name == "vit_base_patch16_224_multi_adapter":
                model = vit_multi_adapter.vit_base_patch16_224_multi_adapter(num_classes=0,
                    global_pool=False, drop_path_rate=0.0, tuning_config=tuning_config)
                model.out_dim=768
            else:
                raise NotImplementedError("Unknown type {}".format(name))
            return model.eval()
        else:
            raise NotImplementedError("Inconsistent model name and model type")

    raise NotImplementedError("Unknown backbone_type: {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]

        if 'resnet' in args['backbone_type']:
            self.model_type = 'cnn'
        else:
            self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        if self.model_type == 'cnn':
            self.backbone(x)['features']
        else:
            return self.backbone(x)

    def forward(self, x):
        if self.model_type == 'cnn':
            x = self.backbone(x)
            out = self.fc(x['features'])
            """
            {
                'fmaps': [x_1, x_2, ..., x_n],
                'features': features
                'logits': logits
            }
            """
            out.update(x)
        else:
            x = self.backbone(x)
            out = self.fc(x)
            out.update({"features": x})

        return out

    def update_fc(self, hash_code_length):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self





class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.W_rand = None

    def init_fc(self, hash_code_length):

        feature_dim = self.feature_dim
        fc = self.generate_fc(feature_dim, hash_code_length).to(self._device)

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def update_fc(self, fc_new):
        del self.fc
        self.fc = fc_new


    def extract_vector(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
 
        out = self.fc(x)
        out.update({"features": x})
        return out




class HashingLayer(nn.Module):
    def __init__(self, in_d, hash_code_length, device, args):
        super(HashingLayer, self).__init__()

        self.in_dim = in_d
        self.hash_code_length = hash_code_length
        self.device = device
        self.gamma = args["rg_hash"]

        self.fc = nn.Linear(self.in_dim, self.hash_code_length, bias=False)

        self.R = self.gamma * torch.eye(self.in_dim).to(self.device)
        self.Q = None  
        self.G = None  
        
    def forward(self, x):
        hash_code = self.fc(x)
        return {'hash_code': hash_code}


