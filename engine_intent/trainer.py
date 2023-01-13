import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.visualization import Visualizer
from .utils import print_info, viz_results, post_process, eval_metrics, PredAnalyst
import torchmetrics
from tqdm import tqdm
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


def do_train(cfg, epoch, model, optimizer, dataloader, device, logger=None, lr_scheduler=None):
    model.train()
    viz = Visualizer(mode='image')
    
    with torch.set_grad_enabled(True):
        for iters, batch in enumerate(tqdm(dataloader)):
#             y_global = batch['target_y']

            if cfg.method == 'rn' or cfg.method == 'cnn_mlp':
                if cfg.model.pred_intent: 
                    pred_intent = model(batch['input_x'].to(device) , batch['obs_image'])
#                 else:
#                     pred_traj = model(batch['input_x'].to(device) , batch['obs_image'])
            
            if cfg.method!='vattn':
                if not cfg.model.pred_intent: # only compute trajectory loss
                    loss=ade_loss(pred_traj, batch['target_y'].to(device))
    #             model.param_scheduler.step()
                    loss_dict = {'loss_traj': loss.item()}
                else: # compute trajectory loss + intent classification loss
#                     loss_traj = ade_loss(pred_traj, batch['target_y'].to(device))
                    loss_intent = torch.nn.CrossEntropyLoss()(pred_intent, batch['pred_intent'].to(device))
#                     loss = loss_traj + loss_intent
                    loss = loss_intent
                    loss_dict = {
#                         'loss_traj': loss_traj.item(), 
                        'loss_intent': loss_intent.item()}
            else:
                loss_dict = {'loss_scene': loss.item()}
#             loss_dict = {k:v.item() for k, v in loss_dict.items()}
            loss_dict['lr'] = optimizer.param_groups[0]['lr']
            # optimize
            optimizer.zero_grad() # avoid gradient accumulate from loss.backward()
            loss.backward()
            
            # loss_dict['grad_norm'] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()

            if cfg.solver.scheduler == 'exp':
                lr_scheduler.step()
            if iters % cfg.print_interval == 0:
                print_info(epoch, model, optimizer, loss_dict, logger)

            if cfg.visualize and iters % max(int(len(dataloader)/5), 1) == 0:
                ret = post_process(cfg, batch['input_x'].to(device), y_global, pred_traj)
                X_global, y_global, pred_goal, pred_traj, dist_traj, dist_goal = ret
                img_path = batch['cur_image_file']
                dist_goal, dist_traj = None, None
                viz_results(viz, X_global, y_global, pred_traj, img_path, dist_goal, dist_traj,
                            bbox_type=cfg.dataset.bbox_type, normalized=False, logger=logger, name='pred_train')
            
            
def do_val(cfg, epoch, model, dataloader, device, logger=None):
    model.eval()
    loss_val, loss_traj_val, loss_intent_val = 0.0, 0.0, 0.0
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader)):
            # y_global = batch['target_y'].to(device)

            if cfg.method == 'rn' or cfg.method == 'cnn_mlp':
                if cfg.model.pred_intent: 
                    pred_intent = model(batch['input_x'].to(device) , batch['obs_image'])
#                 else:
#                     pred_traj = model(batch['input_x'].to(device) , batch['obs_image'])

            if cfg.method!='vattn':
                if not cfg.model.pred_intent: # only compute trajectory loss
                    loss = ade_loss(pred_traj, y_global)
    #             model.param_scheduler.step()
                    loss_dict = {'loss_traj': loss.item()}
                else: # compute trajectory loss + intent classification loss
#                     loss_traj = ade_loss(pred_traj, y_global)
                    loss_intent = torch.nn.CrossEntropyLoss()(pred_intent, batch['pred_intent'].to(device))
#                     loss = loss_traj + loss_intent
                    loss = loss_intent
                    loss_dict = {
#                         'loss_traj': loss_traj.item(), 
                        'loss_intent': loss_intent.item()}
#                     loss_traj_val += loss_traj.item()
                    loss_intent_val += loss_intent.item()
            # compute total loss
            loss_val += loss.item()
            
    loss_val /= (iters + 1)

    if not cfg.model.pred_intent:
        info_str = "loss_val:{:.4f},".format(loss_val)
        info_dict = {'loss_val':loss_val, }
    else:
#         loss_traj_val /= (iters + 1)
        loss_intent_val /= (iters + 1)
#         info_str = "loss_val:{:.4f},\t loss_traj_val:{:.4f},\t loss_intent_val:{:.4f},\t".format(loss_val, loss_traj_val, loss_intent_val)
        info_str = "loss_val:{:.4f},\t loss_intent_val:{:.4f},\t".format(loss_val, loss_intent_val)
        info_dict = {'loss_val':loss_val, 
#                      'loss_traj_val':loss_traj_val, 
                     'loss_intent_val':loss_intent_val
                    }
    if hasattr(logger, 'log_values'):
        logger.info(info_str)
        logger.log_values(info_dict)#, step=epoch)
    else:
        print(info_str)
    return loss_val

def inference(cfg, epoch, model, dataloader, device, logger=None, eval_kde_nll=False, test_mode=False):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    # evaluation metrics
    # metrics = eval_metrics(cfg.test.metrics)

    # prediction tracker 
    analyst = PredAnalyst(cfg) if 'test' in cfg.stats_anl.subsets else None 
    loss_val = 0.0
    ## Collect the test results and test labels:
    acc = torchmetrics.Accuracy()
    f1 = torchmetrics.F1Score(num_classes=cfg.model.num_pedintent_class)
    auc = torchmetrics.AUROC(pos_label=1)
    roc = torchmetrics.ROC(pos_label=1)
    precision = torchmetrics.Precision(num_classes=cfg.model.num_pedintent_class)
    recall = torchmetrics.Recall(num_classes=cfg.model.num_pedintent_class)
    pre_recall = torchmetrics.PrecisionRecallCurve(pos_label=1)

    viz = Visualizer(mode='image')  # for visualizing trajectories
    with torch.set_grad_enabled(False):
        for iters, batch in enumerate(tqdm(dataloader)):
            # y_global = batch['target_y']

            if cfg.method == 'rn' or cfg.method == 'cnn_mlp':
                if cfg.model.pred_intent: 
                    pred_intent = model(batch['input_x'].to(device) , batch['obs_image'])
                    loss_intent = torch.nn.CrossEntropyLoss()(pred_intent, batch['pred_intent'].to(device))
                    info_str = "loss_intent_test:{:.4f},\t".format(loss_intent.item())
                    print(info_str)
                    
                    conf, cls = torch.max(softmax(pred_intent), dim=1)
                    # test_confs[iters] = conf.detach().view(-1, 1).cpu().numpy()
                    # test_labels[iters] = batch['pred_intent'].detach().view(-1, 1).cpu().numpy()
                    acc(cls.detach().view(-1).cpu(), batch['pred_intent'].detach().view(-1).cpu())
                    f1(cls.detach().view(-1).cpu(), batch['pred_intent'].detach().view(-1).cpu())
                    auc(conf.detach().view(-1).cpu(), batch['pred_intent'].detach().view(-1).cpu())
                    # roc.update(conf.detach().view(-1).cpu(), batch['pred_intent'].detach().view(-1).cpu())
                    precision(cls.detach().view(-1).cpu(), batch['pred_intent'].detach().view(-1).cpu())
                    recall(cls.detach().view(-1).cpu(), batch['pred_intent'].detach().view(-1).cpu())
                    # pre_recall.update(conf.detach().view(-1).cpu(), batch['pred_intent'].detach().view(-1).cpu())
                    
                    loss_val += loss_intent.item() # add up the intent loss
                    print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc.compute(), auc.compute(), f1.compute(), precision.compute(), recall.compute()))
                    # auc.reset()
        
    loss_val /= (iters + 1)
    print("loss_intent_test:{:.4f},\t".format(loss_val))
    
    print('acc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc.compute(), f1.compute(), precision.compute(), recall.compute()))

    save_results_path = os.path.join(model_path, '{:.2f}'.format(acc.compute()) + '.yaml')

    # if not os.path.exists(save_results_path):
    results = {'acc': acc.compute(),
                   # 'auc': auc.compute(),
                   'f1': f1.compute(),
                   # 'roc': roc.compute(),
                   'precision': precision.compute(),
                   'recall': recall.compute(),
                   # 'pre_recall_curve': pre_recall.compute()
              }
    return results
