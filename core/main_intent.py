"""The script for intent prediction (only)"""
import time
import os
import sys
sys.path.append(os.path.realpath('.'))
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import shutil
import pickle
from datasets_intent import make_dataloader  ## intent dataloader
from utils.logger import Logger
import logging
import argparse
from configs_intent import cfg  ## intent configurations
from models_intent import make_model
from engine_intent import do_train, do_val, inference

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    
    return args
        
    
def main():
    args = parse_args()
    ## ====================== load configs ======================
    cfg.merge_from_file(args.config_file)  ## merge from unknown list of congigs?
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ## ====================== build model, optimizer and scheduler ======================
    model = make_model(cfg)
    model = model.to(cfg.device)
    print('Model built!')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{cfg.method} model has total {trainable_params} trainable params")

    if cfg.solver.optimizer == 'Adam':
    	optimizer = optim.Adam(model.parameters(), lr=cfg.solver.lr, weight_decay=cfg.solver.weight_decay)
    elif cfg.solver.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg.solver.lr, momentum=cfg.solver.momentum, weight_decay=cfg.solver.weight_decay)
        
    if cfg.solver.scheduler == 'exp':  # exponential schedule
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.solver.GAMMA)
    elif cfg.solver.scheduler == 'plateau':  # Plateau scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.solver.lr_decay_rate, patience=cfg.solver.patience,
                                                            min_lr=5e-06, verbose=1)
    elif cfg.solver.scheduler == 'mslr':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.solver.lr_steps, gamma=cfg.solver.lr_decay_rate)                                 
    print('Schedulers built!')
#     logger = logging.Logger(cfg.method)
    if cfg.use_wandb:
        logger = Logger(cfg.method,
                        cfg,
                        project = cfg.project,
                        viz_backend="wandb"
                        )
    else:
        logger = logging.Logger(cfg.method)
    ## ====================== train, val, test engines ======================
    if cfg.test.inference:  # test mode
        # load test dataloader
        test_dataloader = make_dataloader(cfg, 'test')
        save_checkpoint_dir = os.path.join(cfg.ckpt_dir, cfg.out_dir) 
        epoch = cfg.test.epoch
        model.load_state_dict(torch.load(os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3)))))
        eval_results = inference(cfg, epoch, model, test_dataloader, cfg.device, logger=logger, eval_kde_nll=False)
        print(f'results for the (best) {epoch} epoch: ')

        results_file = os.path.join(save_checkpoint_dir, 'results_{}.pth'.format(str(epoch).zfill(3)))
        if not os.path.exists(results_file):
            pickle.dump(eval_results, open(results_file,'wb'))
    else:
        save_checkpoint_dir = os.path.join(cfg.ckpt_dir, time.strftime("%d%b%Y-%Hh%Mm%Ss"))  # training outputs
        if not os.path.exists(save_checkpoint_dir):
            os.makedirs(save_checkpoint_dir)
        shutil.copy(args.config_file, os.path.join(save_checkpoint_dir, 'configs.yml'))    
        val_loss_best = float('inf')
        epoch_model_best = 0
        # load train / val dataloaders
        train_dataloader = make_dataloader(cfg, 'train')
        val_dataloader = make_dataloader(cfg, 'val')
        for epoch in range(cfg.solver.max_epoch):
            logger.info("Epoch:{}".format(epoch))
            do_train(cfg, epoch, model, optimizer, train_dataloader, cfg.device, logger=logger, lr_scheduler=lr_scheduler)

            val_loss = do_val(cfg, epoch, model, val_dataloader, cfg.device, logger=logger)
            if val_loss_best>val_loss:
                val_loss_best=val_loss
                epoch_model_best=epoch
            torch.save(model.state_dict(), os.path.join(save_checkpoint_dir, 'Epoch_{}.pth'.format(str(epoch).zfill(3))))

            # update LR
            if cfg.solver.scheduler == 'plateau':
                lr_scheduler.step(val_loss)
            elif cfg.solver.scheduler == 'mslr':
                lr_scheduler.step()
            # save the best model based on the lowest validation loss
            with open(os.path.join(save_checkpoint_dir, "log.txt"),"a") as f:
                f.write(f'best val loss: {val_loss_best}, best epoch: {epoch_model_best}')
        print( f'best val loss: {val_loss_best}, best epoch: {epoch_model_best}' )     
if __name__ == '__main__':
    main()
