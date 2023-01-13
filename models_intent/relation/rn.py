"""Simple relation module"""
import os
import sys
sys.path.append(os.path.realpath('.'))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import argparse
from configs import cfg
from .base import BaseModel


class RelationModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # relation module as a set of fully-connected layers
        self.g_fc1 = nn.Linear(2*(cfg.visual.backbone_dim+2)+cfg.motion.enc_hidden_size, 256)
#         self.g_fc2 = nn.Linear(256, 256)
#         self.mot_fc_hidden = nn.Linear(cfg.visual.head_feat_dim, 256)  #  concatenated spatemp-motion feats. with trajectory feats.
        if self.pred_intent:
            self.intent_fc = nn.Linear(cfg.visual.head_feat_dim, self.npc)  
        
    def cvt_coord(self, h, w, i):
        return [(i/h-2)/2., (i%w-2)/2.]

    def forward(self, trajs, imgs,):
        T = trajs.shape[1]
        spat_feats = []
        # (extract) spatial feats.
        for t in range(T):
            spat_feats.append(self.spat_enc(imgs[:, t, ::].to(self.device)))
        spat_feats = torch.stack(spat_feats, dim=2) # (N, C_{in}, D_{in}, H_{in}, W_{in})
        # spatio-temporal feats.
        spatemp_feats = self.temp_enc(spat_feats).squeeze(2)  # torch.Size([N, C, h, w])
        N, C, H, W = spatemp_feats.shape
        spatemp_flat = spatemp_feats.view(N,C,H*W).permute(0,2,1)  # torch.Size([N, h*w, C])
        # optional: append coordinates as `object' identifier
        np_coord_tensor = np.zeros([N, H*W, 2])
        for i in range(H*W):
            np_coord_tensor[:,i,:] = np.array( self.cvt_coord(H, W, i) )
        coord_tensor = torch.from_numpy(np_coord_tensor).float().to(self.device)
        # add coordinates of each `object'(a cell of spatio-temporal feature maps)
        spatemp_flat = torch.cat((spatemp_flat, coord_tensor),2)
        # condition pair-wise spatio-temporal `object' on individual trajectory embedding
        # motion feats.
        traj_feats = self.mot_enc_embed(trajs)
        traj_enc_hidden, _ = self.mot_encoder(traj_feats)
        traj_hidden = traj_enc_hidden[:,-1,:] # last motion hidden states 
        traj_hidden_orig = traj_hidden# the original trajectory embeddnig
        # trajectory embedding
        traj_hidden = torch.unsqueeze(traj_hidden, 1)  #(N, 1, dim)
        traj_hidden = traj_hidden.repeat(1, H*W, 1)  # (N,H*w, dim)
        traj_hidden = torch.unsqueeze(traj_hidden, 2)
        # cast all pairs against each other
        spatemp_i = torch.unsqueeze(spatemp_flat, 1)  # torch.Size([N,  1, h*w, c+2])
        spatemp_i = spatemp_i.repeat(1, H*W, 1, 1)  # torch.Size([N, h*w,h*w, c+2])
        spatemp_j = torch.unsqueeze(spatemp_flat, 2)  # torch.Size([N, h*w,1, c+2])
        spatemp_j = torch.cat((spatemp_j, traj_hidden), 3)
        spatemp_j = spatemp_j.repeat(1, 1, H*W, 1)  # torch.Size([N, h*w,h*w, c+2+dim])
            
        # concatenate all together into pair-wise feats.
        spatemp_full = torch.cat((spatemp_i,spatemp_j),3)  # torch.Size([N, h*w,h*w, 2*(c+2)+dim])
        # spatemp_full = spatemp_full.view(N, (H*W)*(H*W), -1)
        # reshape for passing through network
        spatemp_full = spatemp_full.view(N * (H * W) * (H * W), -1)
        # relational feat. encoding
        x_ = self.g_fc1(spatemp_full)
        x_ = F.relu(x_)
#         x_ = self.g_fc2(x_)
#         x_ = F.relu(x_)
        # x_ = self.g_fc3(x_)
        # x_ = F.relu(x_)
        # x_ = self.g_fc4(x_)
        # x_ = F.relu(x_)
        # reshape again and sum
        x_ = x_.view(N, (H*W)*(H*W), -1)
        # relational feats.
        hidden = x_.sum(1).squeeze() # sum all pairwise relations
        # short cut like residual connection#
        hidden = torch.cat((hidden, traj_hidden_orig), 1)  # Concat motion feats.
        if self.pred_intent:  # with the short cut connection
            x_intent = self.intent_fc(hidden) 
#         hidden = F.relu(self.mot_fc_hidden(hidden))

#         dec_input = self.mot_dec_embed(hidden).unsqueeze(1) 
#         hidden = hidden.unsqueeze(0)
#         outputs = torch.zeros(N, self.pred_len, self.cfg_mot.dec_output_dim).to(self.device)
#         for t in range(self.pred_len):
#             out, hidden = self.mot_decoder(dec_input, hidden)  ## the last hidden of encoder as initial hidden
#             outputs[:,t,:] = out.squeeze(1) 
#             dec_input = self.mot_dec_embed(hidden.squeeze(0)).unsqueeze(1)
        
        if self.pred_intent:
            return x_intent
#         else:
#             return outputs 

class CNN_MLP(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.g_fc1 = nn.Linear(cfg.visual.head_feat_dim, 256)
#         self.mot_fc_hidden = nn.Linear(cfg.visual.head_feat_dim, 256)  #  concatenated spatemp-motion feats. with trajectory feats.
        if self.pred_intent:
            self.intent_fc = nn.Linear(256, self.npc)  
            # self.intent_fc = nn.Linear(cfg.visual.head_feat_dim, self.npc)  

    def forward(self, trajs, imgs):
        N, T = trajs.shape[:2]
        spat_feats = []
        # (extract) spatial feats.
        for t in range(T):
            spat_feats.append(self.spat_enc(imgs[:, t, ::].to(self.device)))
        spat_feats = torch.stack(spat_feats, dim=2) # (N, C_{in}, D_{in}, H_{in}, W_{in})
        # spatio-temporal feats.
        spatemp_feats = self.temp_enc(spat_feats).squeeze(2)  # torch.Size([N, C, h, w])
        """fully connected layers"""
        spatemp_feats = spatemp_feats.view(spatemp_feats.size(0), -1)
        # motion feats.
        traj_feats = self.mot_enc_embed(trajs)
        traj_enc_hidden, _ = self.mot_encoder(traj_feats)
        traj_hidden = traj_enc_hidden[:,-1,:] # last motion hidden states 
        x_ = torch.cat((spatemp_feats, traj_hidden), 1)  # Concat motion feats.
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        if self.pred_intent:
            x_intent = self.intent_fc(x_)
#         hidden = F.relu(self.mot_fc_hidden(x_))
        
#         dec_input = self.mot_dec_embed(hidden).unsqueeze(1) 
#         hidden = hidden.unsqueeze(0)
#         outputs = torch.zeros(N, self.pred_len, self.cfg_mot.dec_output_dim).to(self.device)
#         for t in range(self.pred_len):
#             out, hidden = self.mot_decoder(dec_input, hidden)  ## the last hidden of encoder as initial hidden
#             outputs[:,t,:] = out.squeeze(1) 
#             dec_input = self.mot_dec_embed(hidden.squeeze(0)).unsqueeze(1)
        
        if self.pred_intent:
            return x_intent
#         else:
#             return outputs 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    
    return args
        
if __name__ =='__main__':    
    args = parse_args()
    cfg.merge_from_file(args.config_file)
    model = CNN_MLP(cfg.model).to(cfg.device)
    ### the aspect ratio of the images and the trajectories might need to be the same
    trajs = torch.ones([10,15,4]).to(cfg.device) 
    imgs = torch.ones([10, 15, 3, 216, 384]).to(cfg.device) 
    out = model(trajs, imgs)
    print(out.shape)
