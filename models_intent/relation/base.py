import os
import sys
sys.path.append(os.path.realpath('.'))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import copy
from configs import cfg
from models_intent.motion.motion_predict import EncoderRNN, DecoderRNN


class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm6 = nn.BatchNorm2d(24)
        
    def forward(self, img):
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchNorm5(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchNorm6(x)
        return x

    
class TemporalEncoder(nn.Module):
    def __init__(self, cfg):
        super(TemporalEncoder, self).__init__()
        self.conv1 = nn.Conv3d(cfg.visual.backbone_dim, cfg.visual.backbone_dim, kernel_size=(cfg.input_len, 1, 1), padding=0)
        self.conv2 = nn.Conv2d(cfg.visual.backbone_dim, cfg.visual.backbone_dim, 3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(cfg.visual.backbone_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        return x
    

def SpatialEncoder(cfg):
    if cfg.visual.backbone == 'shallowCNN':
        return ShallowCNN()
    elif cfg.visual.backbone == 'resnet18':
        modules = list(resnet18(pretrained=cfg.visual.backbone_pretrained).children())[:-2]
        modules.append(nn.Conv2d(cfg.visual.backbone_dim, cfg.visual.backbone_dim, 3, stride=2, padding=1))
        return nn.Sequential(*modules)
        # return nn.Sequential(*list(resnet18(pretrained=cfg.visual.backbone_pretrained).children())[:-2])
    else:
        raise Exception('visual encoder not implemented yet')
    
    
class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        # global configs
        self.input_len = cfg.input_len
        self.pred_len = cfg.pred_len
        # (global) intent configs
        self.pred_intent = cfg.pred_intent
        if cfg.pred_intent:
            self.npc = cfg.num_pedintent_class
        # motion configs
        self.cfg_mot = copy.deepcopy(cfg.motion)
        # spatio-temporal feature encoders
        self.spat_enc = SpatialEncoder(cfg) 
        self.temp_enc = TemporalEncoder(cfg)
        # motion encoder
        self.mot_enc_embed = nn.Sequential(nn.Linear(self.cfg_mot.global_input_dim, self.cfg_mot.enc_input_size), nn.ReLU())
        self.mot_encoder = EncoderRNN(self.cfg_mot.enc_input_size, self.cfg_mot.enc_hidden_size, 1) 
        # motion decoder
#         self.mot_dec_embed = nn.Sequential(nn.Linear(self.cfg_mot.enc_hidden_size, self.cfg_mot.dec_input_size), nn.ReLU())
#         self.mot_decoder = DecoderRNN(self.cfg_mot.dec_input_size, self.cfg_mot.dec_hidden_size, self.cfg_mot.dec_output_dim, 1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
