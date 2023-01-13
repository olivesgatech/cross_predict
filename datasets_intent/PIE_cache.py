import os
import sys
sys.path.append('../cross_predict/')
import json
import argparse
import numpy as np
from PIL import Image
import pdb
import pdb
import torch
from torch.utils import data
from torchvision import transforms as tvtfs
import pickle5 as pickle
from datasets_intent.pie_data import PIE
from torchvision.transforms import functional as F
import copy
import glob
import time
from tqdm import tqdm
from configs import cfg
# from utils.box_utils import cxcywh_to_x1y1x2y2

class PIEDataset(data.Dataset):
    def __init__(self, cfg, split, transforms=None):
        self.split = split
        self.root = cfg.dataset.root
        self.cfg = cfg
        if transforms is None:
            normalize = tvtfs.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            self.transforms = tvtfs.Compose([
#                 tvtfs.ToTensor(), # normed tensor will be saved to cache on disk
                                   normalize]) 
        else:
            self.transforms = transforms
        # NOTE: add downsample function
        self.downsample_step = int(30/self.cfg.dataset.fps)
        traj_data_opts = {'fstride': 1,  # the same 
                 'sample_type': 'all',  # the same
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # the same
                 'seq_type': 'crossing',
                 'min_track_size': 76, # the same  # 
                 'random_params': {'ratios': None,
                                 'val_data': True,
                                 'regen_data': True},
#                  'kfold_params': {'num_folds': 5, 'fold': 1}
                         }

        traj_model_opts = {'normalize_bbox': True,  ## is okay for intent predict.
                       'track_overlap': 0.6,    # 0.6 for the PIE dataset
                       'observe_length': cfg.model.input_len,
#                        'predict_length': cfg.,
                       'enc_input_type': ['bbox', ],  # , 'looks'
                       'dec_input_type': [],  # as intention tags
                       'prediction_type': ['activities']
                       }
        imdb = PIE(data_path=self.root)
        
#         traj_model_opts['enc_input_type'].extend(['traffic_bbox_seq', 'traffic_class_seq', 'traffic_obj_id_seq']) 
#         traj_model_opts['prediction_type'].extend([''])
        beh_seq = imdb.generate_data_trajectory_sequence(self.split, **traj_data_opts)

        self.data = self.get_traj_data(beh_seq, **traj_model_opts)

    def __getitem__(self, index):
        obs_bbox = torch.FloatTensor(self.data['obs_bbox'][index])
#         pred_bbox = torch.FloatTensor(self.data['pred_bbox'][index])
        cur_image_file = self.data['obs_image'][index][-1]
        obs_image = self.data['obs_image'][index]
#         pred_image = self.data['pred_image'][index]
        obs_pid = self.data['obs_pid'][index]
#         pred_pid = self.data['pred_pid'][index][0]
#         obs_occlusion = self.data['obs_occlusion'][index]
        pred_intent = torch.LongTensor(self.data['pred_intent'][index])[0][0]
        if self.cfg.model.use_traffic:
            obs_traffic_bbox = self.data['obs_traffic_bbox'][index]
#         pred_traffic_bbox = self.data['pred_traffic_bbox'][index]
            obs_traffic_class = self.data['obs_traffic_class'][index]
#         obs_traffic_oid = self.data['obs_traffic_oid'][index]
        # width, height = Image.open(cur_image_file).size
        
        ret = {'input_x':obs_bbox, 'obs_image_file': obs_image,
#                'obs_traffic_bbox': obs_traffic_bbox, 'obs_traffic_class': obs_traffic_class,
               'obs_pid':obs_pid, 'pred_intent':pred_intent,
#                'obs_obd_speed': obs_obd_speed, 'pred_obd_speed':pred_obd_speed,
#                'obs_gps_coord': obs_gps_coord, 'pred_gps_coord': pred_gps_coord,'target_y':pred_bbox,
                'cur_image_file':cur_image_file}
                    
        ret['timestep'] = int(cur_image_file.split('/')[-1].split('.')[0])
        if self.cfg.model.visual.use_frame:  # load observation frames
            imgs = [] # full frames
            for imp in obs_image:
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0] # frame number
                img_save_folder = os.path.join(self.cfg.dataset.cacher.path, self.cfg.dataset.name, set_id, vid_id)
                img_save_path = os.path.join(img_save_folder, img_name + '.pkl')    
                # if the frame was not saved, save it to disk
                if not os.path.exists(img_save_path): 
                    os.makedirs(img_save_folder, exist_ok=True)
                    img = Image.open(imp).convert('RGB').resize(self.cfg.model.img_size) 
                    img = tvtfs.ToTensor()(img)  # save tensor
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img, fid, pickle.HIGHEST_PROTOCOL)
                else:
                    with open(img_save_path, 'rb') as fid:
                        img = pickle.load(fid)  #  tensor
#                     img = torch.from_numpy(np.asarray(img))
                if self.transforms is not None:
                    img = self.transforms(img)
                imgs.append(img)        
            ret['obs_image'] = torch.stack(imgs)   

        if self.cfg.model.visual.use_scene_states: # load scene states
            w_grid, h_grid = width/cfg.model.Num_WS, height/cfg.model.Num_HS
            # construct scene state labels    
            obs_scene_states, obs_ped_grid = [], [] 
            for (bbox_traffic, bbox_agent) in zip (obs_traffic_bbox, obs_bbox): # iterate each time step    
                centers = (bbox_traffic[:, :2] + bbox_traffic[:, 2:])/2  # centers of traffic objects
                x_indices = np.ceil(centers[:, 0] / w_grid).astype("int")
                y_indices = np.ceil(centers[:, 1] / h_grid).astype("int")
                # scene[0, 0] = 0, we need
                x_indices[x_indices == 0] = 1
                y_indices[y_indices == 0] = 1
                x_indices = x_indices - 1
                y_indices = y_indices - 1
                # construct multi-hot scene state label from other traffic objects
                scene_state = np.zeros((cfg.model.Num_HS, cfg.model.Num_WS)) 
                scene_state[y_indices, x_indices] = 1
                # add state from the target agent
                if self.cfg.dataset.bbox_type=='cxcywh':
                    cx_agent, cy_agent = bbox_agent[:2]  #  agent's center
                else: 
                    raise NotImplementedError
                if self.cfg.dataset.normalize=='zero-one':
                    cx_agent, cy_agent = cx_agent.item()*width, cy_agent.item()*height  # denormalize them
                else:
                    cx_agent, cy_agent = cx_agent.item(), cy_agent.item()
                x_ind_agent = np.ceil(cx_agent/w_grid).astype("int")
                y_ind_agent = np.ceil(cy_agent/h_grid).astype("int") 
                scene_state[y_ind_agent-1, x_ind_agent-1] = 1
                ## multi-class label for the current time step
                scene_state = scene_state.flatten()  
                obs_ped_grid.append(torch.FloatTensor([(x_ind_agent-1)+(y_ind_agent-1)*self.cfg.model.Num_WS]))
                obs_scene_states.append(torch.FloatTensor(scene_state))    
            ret['obs_scene_states'] = torch.stack(obs_scene_states)
            ret['obs_ped_grid'] = torch.stack(obs_ped_grid)
          ## under USE_AGENT_ROI need to add use_traffics objects --> major issues is to load cropped images ... for now only do training on pedestrians only... no contrast on other objects.. just directly load them,
        if self.cfg.model.visual.use_agent_roi:   # use agent's ROI
            roi_agent = self._load_agent_roi(obs_image, obs_pid)
            if self.cfg.dataset.twocrop_tf:   # use two views of data
                ret['obs_image_crops'] = [torch.stack(roi_agent[0]), torch.stack(roi_agent[1])]
            else:
                ret['obs_image_crops'] = torch.stack(roi_agent)
            
            if self.cfg.method == 'supcon':
                roi_agent = self._load_agent_roi(pred_image, pred_pid)
                if self.cfg.dataset.twocrop_tf:
                    ret['pred_image_crops'] = [torch.stack(roi_agent[0]), torch.stack(roi_agent[1])]
                else:
                    ret['pred_image_crops']=torch.stack(roi_agent)
        if self.cfg.model.visual.use_attn:
            ret['obs_look'] = torch.LongTensor(self.data['obs_look'][index])
            ret['pred_look'] = torch.LongTensor(self.data['pred_look'][index])
#         if self.cfg.model.pred_intent: # if predict intent (future act)
#             ret['intent'] = torch.LongTensor(self.data['pred_cross'][index])[int(self.cfg.dataset.ts_intent*self.cfg.dataset.fps)]
        if self.cfg.model.use_traffic: # for graph neural networks
            # loading the time-varying data as list of dictioaries
            traffics, traffic_masks = [], []
            scale = np.array(self.cfg.model.img_size*2)
            for j, (traffic_bbox, traffic_class) in enumerate(zip(obs_traffic_bbox, obs_traffic_class)):
                traffic_dict = {k:[] for k in range(6)}
                traffic_mask_dict = {k:[] for k in range(6)}
                for i in range(len(traffic_class)): # iterating each objext
                    if traffic_class[i] == -1 or traffic_bbox[i].sum() == 0: continue  # invalid object
                    traffic_dict[traffic_class[i]].append(traffic_bbox[i])
                    
                    if self.cfg.model.visual.use_traffic_mask:
                        tr_xmin, tr_ymin, tr_xmax, tr_ymax = cxcywh_to_x1y1x2y2(traffic_bbox[i]*scale).astype(int)
                        obj_xmin, obj_ymin, obj_xmax, obj_ymax = cxcywh_to_x1y1x2y2(obs_bbox[j].numpy()*scale).astype(int)
#                         mask = np.zeros((height, width), np.uint8)  # create a binary mask without I/o operations
                        mask = np.zeros((self.cfg.model.img_size[1], self.cfg.model.img_size[0]), np.uint8) 
                        mask[tr_ymin:tr_ymax,tr_xmin:tr_xmax] = 255
                        mask[obj_ymin:obj_ymax,obj_xmin:obj_xmax] = 255
                        # resize then transform the mask
#                         mask = tvtfs.Compose([tvtfs.Resize((self.cfg.model.img_size[1], self.cfg.model.img_size[0])),
#                                               tvtfs.ToTensor()])(Image.fromarray(mask))
                        mask = tvtfs.ToTensor()(Image.fromarray(mask))
                        # deepcopy to avoid shared memory issues...
                        mask_cp = copy.deepcopy(mask)  
                        del mask  
                        traffic_mask_dict[traffic_class[i]].append(mask_cp)
#                         traffic_mask_dict[traffic_class[i]].append(mask)
            
                for k, v in traffic_dict.items():
                    traffic_dict[k] = np.array(v)
                traffics.append(traffic_dict)
                if self.cfg.model.visual.use_traffic_mask:
                    assert sum([len(v) for v in list(traffic_dict.values())]) == sum([len(v) for v in list(traffic_mask_dict.values())])
                traffic_masks.append(traffic_mask_dict)
            ret['traffics'] = traffics
            if self.cfg.model.visual.use_traffic_mask:
                ret['traffic_masks'] = traffic_masks
        return ret

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])

    def get_traj_tracks(self, dataset, data_types, observe_length, predict_length, overlap, normalize):
        """
        Generates tracks by sampling from pedestrian sequences
        :param dataset: The raw data passed to the method
        :param data_types: Specification of types of data for encoder and decoder. Data types depend on datasets. e.g.
        JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
        :param observe_length: The length of the observation (i.e. time steps of the encoder)
        :param predict_length: The length of the prediction (i.e. time steps of the decoder)
        :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected
        :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
        the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
        :return: A dictinary containing sampled tracks for each data modality
        """
        #  Calculates the overlap in terms of number of frames
#         seq_length = observe_length + predict_length
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check the validity of keys selected by user as data type
        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except:# KeyError:
                raise KeyError('Wrong data type is selected %s' % dt)

        d['image'] = dataset['image']
        d['pid'] = dataset['pid']
        d['resolution'] = [dataset['image_dimension'],]*len(d['pid'])#dataset['resolution']
        num_trks = len(d['image'])
#         leave_out = []
#         for track in d['occlusion']:  # each track is a complete traj, for an agent, next time when it comes to a different key, we will need to get another
#             leave_out.append([])
#             for i in range(0, len(track) - seq_length + 1, overlap_stride): ## sample the track here
#                 if (track[i:i + seq_length][self.downsample_step-1:observe_length:self.downsample_step].count(2)>=observe_length//2): # leave out the track if the agent was fully-occluded across more than half of the frames in this sequence
#                     leave_out[-1].append(i)

        #  Sample tracks from sequneces
        for k in d.keys():
            tracks = []
            for track in d[k]:
                start_idx = len(track) - observe_length - self.cfg.model.time_to_event[1]
                end_idx = len(track) - observe_length - self.cfg.model.time_to_event[0]
                tracks.extend([track[i:i + observe_length] for i in
                                 range(start_idx, end_idx + 1, overlap_stride)])
#                 for i in range(0, len(track) - seq_length + 1, overlap_stride):
#                     if self.cfg.dataset.excl_ocln and (i in lout): continue
#                     tracks.append(track[i:i + seq_length])
            d[k] = tracks
#         d['traffic_bbox_orig'] = (d['traffic_bbox_seq']).copy() # copy.deepcopy # unormed traffic boxes
#         d['traffic_bbox_seq'] = self.convert_normalize_bboxes(d['traffic_bbox_seq'], d['resolution'], 
#                                                   self.cfg.dataset.normalize, self.cfg.dataset.bbox_type)
        
        d['bbox'] = self.convert_normalize_bboxes(d['bbox'], d['resolution'], 
                                                  self.cfg.dataset.normalize, self.cfg.dataset.bbox_type)
        return d

    def convert_normalize_bboxes(self, all_bboxes, all_resolutions, normalize, bbox_type):
        '''input box type is x1y1x2y2 in original resolution'''
        for i in range(len(all_bboxes)):
            if len(all_bboxes[i]) == 0:
                continue
            bbox = np.array(all_bboxes[i])
            # NOTE ltrb to cxcywh
            if bbox_type == 'cxcywh':
                bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
                bbox[..., [0, 1]] += bbox[..., [2, 3]]/2
            # NOTE Normalize bbox
            if normalize == 'zero-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.dataset.min_bbox)[None, :]
                _max = np.array(self.cfg.dataset.max_bbox)[None, :]
                bbox = (bbox - _min) / (_max - _min)
            elif normalize == 'plus-minus-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.cfg.dataset.min_bbox)[None, :]
                _max = np.array(self.cfg.dataset.max_bbox)[None, :]
                bbox = (2 * (bbox - _min) / (_max - _min)) - 1
            elif normalize == 'none':
                pass
            else:
                raise ValueError(normalize)
            all_bboxes[i] = bbox
        return all_bboxes

    def get_data_helper(self, data, data_type):
        """
        A helper function for data generation that combines different data types into a single representation
        :param data: A dictionary of different data types
        :param data_type: The data types defined for encoder and decoder input/output
        :return: A unified data representation as a list
        """
        if not data_type:
            return []
        d = []
        for dt in data_type:
            if dt == 'image':
                continue
            d.append(np.array(data[dt]))
            
        #  Concatenate different data points into a single representation
        if len(d) > 1:
            return np.concatenate(d, axis=2)
        elif len(d) == 1:
            return d[0]
        else:
            return d

    def get_traj_data(self, data, **model_opts):
        """
        Main data generation function for training/testing
        :param data: The raw data
        :param model_opts: Control parameters for data generation characteristics (see below for default values)
        :return: A dictionary containing training and testing data
        """
        
        opts = {
            'normalize_bbox': True,
            'track_overlap': 0.5,
            'observe_length': self.cfg.model.input_len,
            'predict_length': self.cfg.model.pred_len, #0
            'enc_input_type': ['bbox'],
            'dec_input_type': [],
            'prediction_type': ['bbox']
        }
        for key, value in model_opts.items():
            assert key in opts.keys(), 'wrong data parameter %s' % key
            opts[key] = value
  
        observe_length = opts['observe_length']
        data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
        data_tracks = self.get_traj_tracks(data, data_types, observe_length,
                                      opts['predict_length'], opts['track_overlap'],
                                      opts['normalize_bbox'])
        obs_slices = {}
        pred_slices = {}

        #  Generate observation/prediction sequences from the tracks
        for k in data_tracks.keys():
            obs_slices[k] = []
            pred_slices[k] = []
            # NOTE: Add downsample function
            down = self.downsample_step
            obs_slices[k].extend([d[down-1:observe_length:down] for d in data_tracks[k]])
            pred_slices[k].extend([d[observe_length+down-1::down] for d in data_tracks[k]])
        
        ret =  {'obs_image': obs_slices['image'],
                'obs_pid': obs_slices['pid'],
                'obs_resolution': obs_slices['resolution'],
                'obs_bbox': np.array(obs_slices['bbox']), 
                'pred_intent': obs_slices['activities'],
#                 'obs_cross': obs_slices['cross'],
#                 'obs_look': (obs_slices['looks']),
#                 'obs_trbbox_unorm': obs_slices['traffic_bbox_orig'],
#                 'obs_traffic_bbox': obs_slices['traffic_bbox_seq'],
#                 'obs_traffic_class': obs_slices['traffic_class_seq'],
#                 'obs_traffic_oid': obs_slices['traffic_obj_id_seq'], 
                'pred_image': pred_slices['image'],
                'pred_pid': pred_slices['pid'],
                'pred_bbox': np.array(pred_slices['bbox']), 
#                 'pred_cross': pred_slices['cross'],
#                 'pred_look': (pred_slices['looks']),
#                 'obs_occlusion': obs_slices['occlusion'],
                }
        
        return ret

    def _load_agent_roi(self, _image, _pid):
        img_crops = [] # cropped objects 
        img_crops_copy = [] # additional crops of objects
        for imp, p in zip(_image, _pid):
            set_id = imp.split('/')[-3]
            vid_id = imp.split('/')[-2]
            img_name = imp.split('/')[-1].split('.')[0]
            img_save_folder = os.path.join(self.cfg.dataset.root, f'images_crop/{cfg.model.visual.crop_type}', set_id, vid_id) 
            img_save_path = os.path.join(img_save_folder, img_name + '-' + p[0] + '.pkl')
            try:
                with open(img_save_path, 'rb') as fid:
                    img_crop=pickle.load(fid)  ## numpy array
            except:   ##TODO: resolve the corrupt image/crops 
                print(f'{img_save_path} can not be loaded')
                img_crop = (np.zeros((self.cfg.model.visual.pad_size,self.cfg.model.visual.pad_size, 3)) * 255).astype(np.uint8)
            if self.transforms is not None:
                img_crop = self.transforms(Image.fromarray(img_crop))
            
            if self.cfg.dataset.twocrop_tf:  # two views of data
                img_crops.append(img_crop[0])
                img_crops_copy.append(img_crop[1])     
            else:
                #deepcopy to avoid shared memory issues...
#                 img_crops.append(img_crop)   
                img_crop_cp = copy.deepcopy(img_crop)  
                del img_crop  
                img_crops.append(img_crop_cp)   

        if self.cfg.dataset.twocrop_tf:
            return [img_crops, img_crops_copy]
        return img_crops
    
#     def _behavioral_stats(self, tag=''):
#         """check statistics of behavioral tags"""
#         stats = [sum(item) for item in self.beh_seq[tag]]
#         idx = np.nonzero(stats)[0]
#         print(f'total {tag}: {len(idx)}')  # e.g., total looks: 478
#         print(f'total non-{tag}: {len(stats)-len(idx)}')  # e.g., total non-looks: 388
