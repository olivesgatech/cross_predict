"""visualization utils"""
import os
from PIL import Image
import numpy as np
import cv2
import torch
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    from .box_utils import cxcywh_to_x1y1x2y2
except:
    from box_utils import cxcywh_to_x1y1x2y2
class Visualizer():
    def __init__(self, mode='image'):
        self.mode = mode
        if self.mode == 'image':
            self.img = None
        elif self.mode == 'plot':
            self.fig, self.ax = None, None
        else:
            raise NameError(mode)
            
    def initialize(self, img_path=None):
        if self.mode == 'image':
            self.img = np.array(Image.open(img_path))
            self.H, self.W, self.CH = self.img.shape
        elif self.mode == 'plot':
            self.fig, self.ax = plt.subplots()
    
    def visualize(self, 
                  inputs, 
                  id_to_show=0,
                  normalized=False, 
                  bbox_type='x1y1x2y2',
                  color=(255,0,0), 
                  thickness=4, 
                  radius=5,
                  label=None,  
                  viz_type='point', 
                  viz_time_step=None):
        if viz_type == 'bbox':
            self.viz_bbox_trajectories(inputs, normalized=normalized, bbox_type=bbox_type, color=color, viz_time_step=viz_time_step)
        elif viz_type == 'point':
            self.viz_point_trajectories(inputs, color=color, label=label, thickness=thickness, radius=radius)
        elif viz_type == 'distribution':
            self.viz_distribution(inputs, id_to_show, thickness=thickness, radius=radius)
    
    def clear(self):
        plt.close()
        # plt.cla()
        # plt.clf()
        self.fig.clear()
        self.ax.clear()
        del self.fig, self.ax
    
    def save_plot(self, fig_path, clear=True):
        self.ax.set_xlabel('x [m]', fontsize=12)
        self.ax.set_ylabel('y [m]', fontsize=12)
        self.ax.legend(fontsize=12)
        plt.savefig(fig_path)
        if clear:
            self.clear()
            
    def viz_point_trajectories(self, points, color=(255,0,0), label=None, thickness=4, radius=5):
        '''
        points: (T, 2) or (T, K, 2)
        '''
        if self.mode == 'image':
            # plot traj on image
            if len(points.shape) == 2:
                points = points[:, None, :]
            T, K, _ = points.shape
            points = points.astype(np.int32)
            for k in range(K):
#                 pdb.set_trace()
                cv2.polylines(self.img, [points[:, k, :]], isClosed=False, color=color, thickness=thickness)
                    
                for t in range(T):
                    cv2.circle(self.img, tuple(points[t, k, :]), color=color, radius=radius, thickness=-1)
        elif self.mode == 'plot':
            # plot traj in matplotlib 
            # pdb.set_trace()
            if len(points.shape) == 2:
                self.ax.plot(points[:, 0], points[:, 1], '-o', color=color, label=label)
            elif len(points.shape) == 3:
                # multiple traj as (T, K, 2)
                for k in range(points.shape[1]):
                    label = label if k == 0 else None
                    self.ax.plot(points[:, k, 0], points[:, k, 1], '-', color=color, label=label)
            else:
                raise ValueError('points shape wrong:', points.shape)
            self.ax.axis('equal')

    def draw_single_bbox(self, bbox, color=None):
        '''
        img: a numpy array
        bbox: a list or 1d array or tensor with size 4, in x1y1x2y2 format
        
        '''
        
        if color is None:
            color = np.random.rand(3) * 255
        cv2.rectangle(self.img, (int(bbox[0]), int(bbox[1])), 
                    (int(bbox[2]), int(bbox[3])), color, 2)
            
            
    def viz_bbox_trajectories(self, bboxes, normalized=False, bbox_type='x1y1x2y2', color=None, thickness=4, radius=5, viz_time_step=None):
        '''
        bboxes: (T,4) or (T, K, 4)
        '''
        if len(bboxes.shape) == 2:
            bboxes = bboxes[:, None, :]
#         pdb.set_trace()
        if normalized:
            bboxes[:,:,[0, 2]] *= self.W
            bboxes[:,:,[1, 3]] *= self.H
        if bbox_type == 'cxcywh':
            bboxes = cxcywh_to_x1y1x2y2(bboxes)
        elif bbox_type == 'x1y1x2y2':
            pass
        elif bbox_type == 'xywh':  # original data type...
            raise NotImplementedError
        else:
            raise ValueError(bbox_type)
        bboxes = bboxes.astype(np.int32)
        T, K, _ = bboxes.shape
        
        # also draw the center points
        center_points = (bboxes[..., [0, 1]] + bboxes[..., [2, 3]])/2 # (T, K, 2)
        self.viz_point_trajectories(center_points, color=color, thickness=thickness, radius=radius)

        # draw way point every several frames, just to make it more visible
        if viz_time_step:
            bboxes = bboxes[viz_time_step, :]
            T = bboxes.shape[0]
        for t in range(T):
            for k in range(K):
                self.draw_single_bbox(bboxes[t, k, :], color=color)
