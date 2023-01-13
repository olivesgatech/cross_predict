import os
import sys
sys.path.append('../cross_predict/')
import json
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as tvtfs
from datasets_intent.jaad_data import JAAD
import argparse
import pickle5 as pickle
from configs import cfg
import copy
import glob
import time

class JAADDataset(data.Dataset):
    def __init__(self, cfg, split, transforms=None):
        self.split = split
        self.root = cfg.dataset.root
        self.cfg = cfg
        if transforms is None:
            normalize = tvtfs.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
            self.transforms = tvtfs.Compose([
                # tvtfs.ToTensor(), 
                                   normalize]) 
        else:
            self.transforms = transforms
        data_opts = {'fstride': 1,  # the same 
                 'sample_type': 'all', # the same
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # the same
                 'seq_type': 'crossing',
                 'min_track_size': 76, # the same
                 'random_params': {'ratios': None,
                                 'val_data': True,
                                 'regen_data': True},
#                  'kfold_params': {'num_folds': 5, 'fold': 1}
                    }
        traj_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.8,    # 0.6 for the PIE dataset
                       'observe_length': cfg.model.input_len,
                       # 'predict_length': cfg.model.pred_len,
                       'enc_input_type': ['bbox', ],
                       'dec_input_type': [], #['intention_prob', 'obd_speed'],
                       'prediction_type': ['activities'] 
                       }
        self.downsample_step = int(30/self.cfg.dataset.fps)
        imdb = JAAD(data_path=self.root)
        beh_seq = imdb.generate_data_trajectory_sequence(self.split, **data_opts)
        self.data = self.get_data(beh_seq, **traj_model_opts)

    def __getitem__(self, index):
        obs_bbox = torch.FloatTensor(self.data['obs_bbox'][index])
        # pred_bbox = torch.FloatTensor(self.data['pred_bbox'][index])
        cur_image_file = self.data['obs_image'][index][-1]
        # pred_resolution = torch.FloatTensor(self.data['pred_resolution'][index])
        obs_image = self.data['obs_image'][index]
        # pred_image = self.data['pred_image'][index]
        obs_pid = self.data['obs_pid'][index]
        # pred_pid = self.data['pred_pid'][index][0]
        # obs_occlusion = self.data['obs_occlusion'][index]
        pred_intent = torch.LongTensor(self.data['pred_intent'][index])[0][0]
        ret = {'input_x':obs_bbox, 'obs_image_file': obs_image,
               'obs_pid':obs_pid, 'pred_intent':pred_intent,
               # 'pred_pid':pred_pid, 'obs_occlusion': obs_occlusion,
               # 'target_y':pred_bbox, 
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
                    # img = torch.from_numpy(np.asarray(img))
                if self.transforms is not None:
                    img = self.transforms(img)
                imgs.append(img)        
            ret['obs_image'] = torch.stack(imgs)   
            
        if self.cfg.model.visual.use_agent_roi:
            imgs, img_crops = [], [] # full frames, cropped objects 
            for imp, p in zip(obs_image, obs_pid):
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                img_save_folder = os.path.join(self.cfg.dataset.root, f'images_crop/{cfg.model.CROP_TYPE}', set_id, vid_id) 
                img_save_path = os.path.join(img_save_folder, img_name + '-' + p[0] + '.pkl')
                with open(img_save_path, 'rb') as fid:
                    img_crop=pickle.load(fid)  ## numpy array
                if self.transforms is not None:
                    img_crop = self.transforms(Image.fromarray(img_crop))
                img_crops.append(img_crop)
        
            ret['obs_image_crops']=torch.stack(img_crops)
        
        # if self.cfg.model.pred_intent: # if predict intent (future act)
        #     ret['intent'] = torch.LongTensor(self.data['pred_cross'][index])[int(self.cfg.dataset.ts_intent*self.cfg.dataset.fps)]

        return ret

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])
        
    def get_tracks(self, dataset, data_types, observe_length, predict_length, overlap, normalize):
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
        # seq_length = observe_length + predict_length
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check the validity of keys selected by user as data type
        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except:
                raise KeyError('Wrong data type is selected %s' % dt)

        d['image'] = dataset['image']
        d['pid'] = dataset['pid']
        d['resolution'] = [dataset['image_dimension'],]*len(d['pid'])#dataset['resolution']

        # # leave_out = []
        # for track in d['occlusion']:  # each track is a complete traj, for an agent, next time when it comes to a different key, we will need to get another
        #     leave_out.append([])
        #     for i in range(0, len(track) - seq_length + 1, overlap_stride): ## sample the track here
        #         if (track[i:i + seq_length][self.downsample_step-1:observe_length:self.downsample_step].count(2)>=observe_length//2): # leave out the track if the agent was fully-occluded across more than half of the frames in this sequence
        #             leave_out[-1].append(i)
        #  Sample tracks from sequneces
        for k in d.keys():
            tracks = []
            # for track in d[k]:
            #     tracks.extend([track[i:i + seq_length] for i in
            #                 range(0, len(track) - seq_length + 1, overlap_stride)])
            for track in d[k]:
                start_idx = len(track) - observe_length - self.cfg.model.time_to_event[1]
                end_idx = len(track) - observe_length - self.cfg.model.time_to_event[0]
                tracks.extend([track[i:i + observe_length] for i in
                                 range(start_idx, end_idx + 1, overlap_stride)])
                # for i in range(0, len(track) - seq_length + 1, overlap_stride):
                #     if self.cfg.dataset.excl_ocln and (i in lout): continue
                #     tracks.append(track[i:i + seq_length])
            d[k] = tracks

        #  Normalize tracks using FOL paper method, 
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

    def get_data(self, data, **model_opts):
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
            'predict_length': self.cfg.model.pred_len,
            'enc_input_type': ['bbox'],
            'dec_input_type': [],
            'prediction_type': ['bbox']
        }
        for key, value in model_opts.items():
            assert key in opts.keys(), 'wrong data parameter %s' % key
            opts[key] = value

        observe_length = opts['observe_length']
        data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
        data_tracks = self.get_tracks(data, data_types, observe_length,
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
                'pred_image': pred_slices['image'],
                'pred_pid': pred_slices['pid'],
                # 'pred_resolution': pred_slices['resolution'],
                'obs_bbox': np.array(obs_slices['bbox']),
                'pred_intent': obs_slices['activities'],
#                 'flow_input': obs_slices['flow'],
                'pred_bbox': np.array(pred_slices['bbox']),
                # 'obs_occlusion': obs_slices['occlusion'],
                # 'obs_cross': obs_slices['cross'],
                # 'pred_cross': pred_slices['cross'],
                'model_opts': opts}
        
        return ret

    def get_path(self,
                 file_name='',
                 save_folder='models',
                 dataset='pie',
                 model_type='trajectory',
                 save_root_folder='data/'):
        """
        A path generator method for saving model and config data. It create directories if needed.
        :param file_name: The actual save file name , e.g. 'model.h5'
        :param save_folder: The name of folder containing the saved files
        :param dataset: The name of the dataset used
        :param save_root_folder: The root folder
        :return: The full path for the model name and the path to the final folder
        """
        save_path = os.path.join(save_root_folder, dataset, model_type, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return os.path.join(save_path, file_name), save_path
    
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Dataloader")
#     parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
#     parser.add_argument(
#         "opts",
#         help="Modify config options using the command-line",
#         default=None,
#         nargs=argparse.REMAINDER,
#     )
    args = parser.parse_args() 
    return args
if __name__=='__main__':
    args = parse_args()   
    print(args)
    cfg.merge_from_file(args.config_file)
    dataset = JAADDataset(cfg, split='train')    
    for data in dataset:
        print(data.keys())
        break
