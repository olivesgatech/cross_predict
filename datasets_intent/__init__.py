import dill
import collections.abc 
from torch.utils.data._utils.collate import default_collate
from .PIE_cache import PIEDataset
from .JAAD_cache import JAADDataset
from torch.utils.data import DataLoader, get_worker_info
import copy

_DATA_LAYERS = {
    'PIE': PIEDataset,
    'JAAD': JAADDataset
 }

def make_dataset(cfg, split, transforms):
    try:
        data_layer = _DATA_LAYERS[cfg.dataset.name]
    except:
        raise NameError("Unknown dataset:{}".format(cfg.dataset.name))
    
    return data_layer(cfg, split, transforms)

def make_dataloader(cfg, split='train',transforms=None):
    if split == 'test':
        batch_size = cfg.test.batch_size
    else:
        batch_size = cfg.solver.batch_size
    dataloader_params ={
            "batch_size": batch_size,
            "shuffle":split == 'train',
            "num_workers": cfg.dataloader.num_workers,
            "collate_fn": collate_dict,
            "pin_memory":True
            }
    
    dataset = make_dataset(cfg, split, transforms)
    if len(dataset)%cfg.solver.batch_size != 0: dataloader_params['drop_last'] = True
    if len(dataset)%cfg.solver.batch_size == 1: dataloader_params['drop_last'] = True
#     if cfg.model.visual.use_traffic_mask:
#         import torch.multiprocessing
#         torch.multiprocessing.set_sharing_strategy('file_system')
    dataloader = DataLoader(dataset, **dataloader_params)
    
    print("{} dataloader size: {}".format(split, len(dataloader)))
    return dataloader

    
def collate_dict(batch):
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    collate_batch = {}
    for key in list(elem.keys()):
        if elem[key] is None:
            collate_batch[key] = None
        elif key == 'traffics':
            traffics = []
            for b in batch:
                traffics += b['traffics'],
            collate_batch['traffics'] = traffics
        elif key == 'traffic_masks':
            traffic_masks = []
            for b in batch:
                traffic_masks += b['traffic_masks'],
            collate_batch['traffic_masks'] = traffic_masks
#             del traffic_masks
        else:
#             neighbor_dict = default_collate([b[key] for b in batch])
#             collate_batch[key] = dill.dumps(neighbor_dict) if get_worker_info() else neighbor_dict
            collate_batch[key] = default_collate([b[key] for b in batch])
    return collate_batch


# def collate_dict(batch):
#     batch_cp=copy.deepcopy(batch)
#     del batch
#     if len(batch_cp) == 0:
#         return batch_cp
#     elem = batch_cp[0]
#     if elem is None:
#         return None
#     collate_batch = {}
#     for key in list(elem.keys()):
#         if elem[key] is None:
#             collate_batch[key] = None
#         elif key == 'traffics':
#             traffics = []
#             for b in batch_cp:
#                 traffics += b['traffics'],
#             collate_batch['traffics'] = traffics
#         elif key == 'traffic_masks':
#             traffic_masks = []
#             for b in batch_cp:
#                 traffic_masks += b['traffic_masks'],
#             collate_batch['traffic_masks'] = traffic_masks
#             del traffic_masks
#         else:
# #             neighbor_dict = default_collate([b[key] for b in batch_cp])
# #             collate_batch[key] = dill.dumps(neighbor_dict) if get_worker_info() else neighbor_dict
#             collate_batch[key] = default_collate([b[key] for b in batch_cp])
#     return collate_batch
                
    
if __name__=='__main__':
    
    train_dataloader = make_dataloader(cfg, 'train')
    
    print()
    
    
