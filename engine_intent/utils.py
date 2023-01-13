import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as tvtfs
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math


class PredAnalyst:
    def __init__(self, cfg):
        """Tracks and analyzes statistics of prediction"""
        self.sampleID_list = [] # the unique sample identity (the current observation timestep)
        self.itemID_list = [] # the unique item(dataset) indices
        self.class_list = []
        self.distance_list = []  # radial distance to the camera
        self.pred_lists = {}
        self.cfg = cfg
        for name in cfg.test.metrics:
            self.pred_lists[name] = []

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def eval_metrics(names: list):
    metrics = {}
    for name in names:
        metrics[name] = AverageMeter(name)
    return metrics

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.LR
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.MAX_EPOCH)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def print_info(epoch, model, optimizer, loss_dict, logger):
    info = "Epoch:{},\t lr:{:6},\t".format(epoch, optimizer.param_groups[0]['lr']) 
    for k in loss_dict.keys():
        info += " {}:{:.4f},\t".format(k, loss_dict[k]) 
    if 'grad_norm' in loss_dict:
        info += ", \t grad_norm:{:.4f}".format(loss_dict['grad_norm'])
    
    if hasattr(logger, 'log_values'):
        logger.info(info)
        logger.log_values(loss_dict)#, step=max_iters * epoch + iters)
    else:
        print(info)

def viz_results(viz, 
                X_global, 
                y_global, 
                pred_traj, 
                img_path, 
                dist_goal, 
                dist_traj,
                bbox_type='cxcywh',
                normalized=True,
                logger=None, 
                name=''):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    id_to_show = np.random.randint(pred_traj.shape[0])

    # 1. initialize visualizer
    viz.initialize(img_path[id_to_show])

    # 2. visualize point trajectory or box trajectory
    if y_global.shape[-1] == 2:
        viz.visualize(pred_traj[id_to_show], color=(0, 1, 0), label='pred future', viz_type='point')
        viz.visualize(X_global[id_to_show], color=(0, 0, 1), label='past', viz_type='point')
        viz.visualize(y_global[id_to_show], color=(1, 0, 0), label='gt future', viz_type='point')
    elif y_global.shape[-1] == 4:
        T = X_global.shape[1]
        viz.visualize(pred_traj[id_to_show], color=(0, 255., 0), label='pred future', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])
        viz.visualize(X_global[id_to_show], color=(0, 0, 255.), label='past', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=list(range(0, T, 3))+[-1])
        viz.visualize(y_global[id_to_show], color=(255., 0, 0), label='gt future', viz_type='bbox', 
                      normalized=normalized, bbox_type=bbox_type, viz_time_step=[-1])        

    # 3. optinaly visualize GMM distribution
    if hasattr(dist_goal, 'mus') and viz.mode == 'plot':
        dist = {'mus':dist_goal.mus.numpy(), 'log_pis':dist_goal.log_pis.numpy(), 'cov': dist_goal.cov.numpy()}
        viz.visualize(dist, id_to_show=id_to_show, viz_type='distribution')
    
    # 4. get image. 
    if y_global.shape[-1] == 2:
        viz_img = viz.plot_to_image(clear=True)
    else:
        viz_img = viz.img

    if hasattr(logger, 'log_image'):
        logger.log_image(viz_img, label=name)

def post_process(cfg, X_global, y_global, pred_traj, pred_goal=None, dist_traj=None, dist_goal=None):
    '''post process the prediction output'''
    if len(pred_traj.shape) == 4:
        batch_size, T, K, dim = pred_traj.shape
    else: # no distribution prediction
        batch_size, T, dim = pred_traj.shape
    X_global = X_global.detach().to('cpu').numpy()
    y_global = y_global.detach().to('cpu').numpy()
#     if pred_goal is not None:
#         pred_goal = pred_goal.detach().to('cpu').numpy()
    pred_traj = pred_traj.detach().to('cpu').numpy()
    
#     if hasattr(dist_traj, 'mus'):
#         dist_traj.to('cpu')
#         dist_traj.squeeze(1)
#     if hasattr(dist_goal, 'mus'):
#         dist_goal.to('cpu')
#         dist_goal.squeeze(1)
    if dim == 4:
        # BBOX: denormalize and change the mode
        _min = np.array(cfg.dataset.min_bbox)[None, None, :] # B, T, dim
        _max = np.array(cfg.dataset.max_bbox)[None, None, :]
        if cfg.dataset.normalize == 'zero-one':
            if pred_goal is not None:
                pred_goal = pred_goal * (_max - _min) + _min
            pred_traj = pred_traj * (_max - _min) + _min
            y_global = y_global * (_max - _min) + _min
            X_global = X_global * (_max - _min) + _min
        elif cfg.dataset.normalize == 'plus-minus-one':
            if pred_goal is not None:
                pred_goal = (pred_goal + 1) * (_max - _min)/2 + _min
            pred_traj = (pred_traj + 1) * (_max[None,...] - _min[None,...])/2 + _min[None,...]
            y_global = (y_global + 1) * (_max - _min)/2 + _min
            X_global = (X_global + 1) * (_max - _min)/2 + _min
        elif cfg.dataset.normalize == 'none':
            pass
        else:
            raise ValueError()

        # NOTE: convert distribution from cxcywh to image resolution x1y1x2y2
        if hasattr(dist_traj, 'mus') and cfg.dataset.normalize != 'none':
        
            _min = torch.FloatTensor(cfg.dataset.min_bbox)[None, None, :].repeat(batch_size, T, 1) # B, T, dim
            _max = torch.FloatTensor(cfg.dataset.max_bbox)[None, None, :].repeat(batch_size, T, 1)
            zeros = torch.zeros_like(_min[..., 0])
            
            if cfg.dataset.normalize == 'zero-one':
                A = torch.stack([torch.stack([(_max-_min)[..., 0], zeros, zeros, zeros], dim=-1),
                                torch.stack([zeros, (_max-_min)[..., 1], zeros, zeros], dim=-1),
                                torch.stack([zeros, zeros, (_max-_min)[..., 2], zeros], dim=-1),
                                torch.stack([zeros, zeros, zeros, (_max-_min)[..., 3]], dim=-1),
                                ], dim=-2)
                b = torch.tensor(_min)
            elif cfg.dataset.normalize == 'plus-minus-one':
                A = torch.stack([torch.stack([(_max-_min)[..., 0]/2, zeros, zeros, zeros], dim=-1),
                                torch.stack([zeros, (_max-_min)[..., 1]/2, zeros, zeros], dim=-1),
                                torch.stack([zeros, zeros, (_max-_min)[..., 2]/2, zeros], dim=-1),
                                torch.stack([zeros, zeros, zeros, (_max-_min)[..., 3]/2], dim=-1),
                                ], dim=-2)
                b = torch.stack([(_max+_min)[..., 0]/2, (_max+_min)[..., 1]/2, (_max+_min)[..., 2]/2, (_max+_min)[..., 3]/2],dim=-1)
            try:
                traj_mus = torch.matmul(A.unsqueeze(2), dist_traj.mus.unsqueeze(-1)).squeeze(-1) + b.unsqueeze(2)
                traj_cov = torch.matmul(A.unsqueeze(2), dist_traj.cov).matmul(A.unsqueeze(2).transpose(-1,-2))
                goal_mus = torch.matmul(A[:, 0:1, :], dist_goal.mus.unsqueeze(-1)).squeeze(-1) + b[:, 0:1, :]
                goal_cov = torch.matmul(A[:, 0:1, :], dist_goal.cov).matmul(A[:,0:1,:].transpose(-1,-2))
            except:
                raise ValueError()

            dist_traj = GMM4D.from_log_pis_mus_cov_mats(dist_traj.input_log_pis, traj_mus, traj_cov)
            dist_goal = GMM4D.from_log_pis_mus_cov_mats(dist_goal.input_log_pis, goal_mus, goal_cov)
    return X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal


def viz_attn_map(normed_img, output, label, input_shape, ):
    """ 
    visualizing attention map
    return: fused visualizations
    """
#     pdb.set_trace()
    segmentation = F.interpolate(output['segmentation'].detach().cpu(), size=input_shape, mode='bilinear', align_corners=False)   # (B, num_classes+1, img_h, img_w)
    segmentation_pixel_classes = torch.argmax(segmentation, dim=1)   # (B, img_h, img_w) 
#     pdb.set_trace()
    hue = (torch.argmax(segmentation, dim=1).float() + 0.5)/(segmentation.shape[1])  # normalizing the hue range to [0,1] / torch.Size([B, img_h, img_w]) 
    inv_normalize = tvtfs.Compose([tvtfs.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                     tvtfs.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                                   ])
    ## perform batch-wise attn vis
    for i in range(len(label)):
        # use the pedestrian grid location first 
#         segmentation_pixel_tp = (segmentation_pixel_classes[i] == (label[i]+1)).int()  # the single-channel mask of the segmap for the specified grid response; (img_h, img_w)
        segmentation_pixel_tp = 0
        for lbl in torch.nonzero(label[i]):
            segmentation_pixel_tp += (segmentation_pixel_classes[i] == lbl.item()+1).int()
                                 
        img = inv_normalize(normed_img[i]).unsqueeze(0) 
        gs_im = img.mean(1)  # average across channels (1, img_h, img_w)
        gs_min = gs_im.min()
        gs_max = torch.max((gs_im-gs_min))  
        gs_im = (gs_im - gs_min)/gs_max
        hsv_im_tp = torch.stack((hue[i].unsqueeze(0).float(), segmentation_pixel_tp.unsqueeze(0).float(), gs_im.float()), -1)
        im_tp = hsv_to_rgb(hsv_im_tp.numpy())
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.imshow(np.squeeze(im_tp, axis=0))
#         axs.set_title(f'ground-truth grid: {label[i].detach().item()}')
        axs.axis('off')
        save_path=os.path.join('random', f'scene_state_{label[i].sum().detach().item()}')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'batch_{i}.png'), dpi=100)
    
    