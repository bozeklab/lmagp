"""
Script for generating attention heatmaps from a transformer-based model.

This script takes a configuration file and a model checkpoint, applies the model to the provided test dataset, and generates attention rollout heatmaps for each sample. 
The heatmaps are saved as small thumbnails images and pickled matrices.

Each sample in the dataset should be a python dict with the following keys:
- features: The sequence of features vectors itself.
- reduced_size: A tuple `(number_of_tiles_h, number_of_tiles_w)` with the size of the WSI after tiling.
- indices: an array of indices `(j,i)` that indicates the indices of the patches (feature vectors) in the tiled WSI. Should have the same number of elements as `features`

Usage:
$ python -cp path/to/config -cn config_name \
    +hydra.run.dir=outputs/attention_heatmaps/${now:%Y-%m-%d}/${now:%H-%M-%S} \
    +checkpoint=path/to/checkpoint.ckpt

"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import hydra
from hydra_zen import instantiate 
from omegaconf import OmegaConf, ListConfig
import torch
from models import LitModel
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import pickle

def attention_rollout(attentions, discard_ratio=0.5, head_fusion='mean', residual=True, i=0, j=0):
    with torch.no_grad():
        if head_fusion == "mean":
            attention_heads_fused = attentions[i].mean(axis=1)
        elif head_fusion == "max":
            attention_heads_fused = attentions[i].max(axis=1)[0]
        elif head_fusion == "min":
            attention_heads_fused = attentions[i].min(axis=1)[0]
        else:
            raise "Attention head fusion type not supported"
        
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
        indices = indices[indices != 0]
        flat[..., indices] = 0

        if residual:   
            I = torch.eye(attention_heads_fused.size(-1))
            attention_heads_fused = (attention_heads_fused + 1.0 * I) / 2
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1)

        if i == j:
            return attention_heads_fused
        elif i > j:
            return torch.bmm(attention_heads_fused, attention_rollout(attentions, discard_ratio, head_fusion, residual, i - 1, j))

def attention_rollout_base_transformer_encoder(attentions, discard_ratio=0.75, head_fusion='max'):
    return torch.bmm(
        attention_rollout(attentions, discard_ratio, head_fusion, residual=True, i=12, j=1),
        attention_rollout([np.insert(attentions[0], 0, 0, axis=-2)], discard_ratio, head_fusion, residual=False, i=0, j=0)
    )

def load_lit_model_from_cfg(cfg):
    module_dict = dict(cfg.module)
    _ = module_dict.pop('_target_')
    module_kwargs_conf = OmegaConf.create(module_dict)
    module_kwargs = instantiate(module_kwargs_conf)
    lit_model = LitModel.load_from_checkpoint(cfg.checkpoint, **module_kwargs)
    return lit_model

@hydra.main(config_path=None)
def task(cfg):
    lit_model = load_lit_model_from_cfg(cfg).eval().cuda()
    
    if cfg.get('validation'):
        test_dl_cfg = cfg.datamodule.val_dataloader
        if isinstance(test_dl_cfg, ListConfig):
            test_dl_cfg = test_dl_cfg[0]
        test_ds_dict = dict(test_dl_cfg.dataset)
    else:
        test_ds_dict = dict(cfg.datamodule.test_dataloader.dataset)
        
    _ = test_ds_dict.pop('transform')
    test_ds_conf = OmegaConf.create(test_ds_dict)
    test_ds = instantiate(test_ds_conf)
    rollout_discard_ratio = dict(cfg).pop('rollout_discard_ratio')

    for fn, lbl in test_ds:
        with open(fn, 'rb') as fp:
            sample = pickle.load(fp)
        features_batch = sample['features'].cuda().unsqueeze(0)
        idcs = sample['indices']
        reduced_size = sample['reduced_size']

        with torch.no_grad():
            y = lit_model(features_batch, output_attentions=True)
        
        if isinstance(y, tuple):
            attentions = y[1]
        else:
            attentions = y['attentions']

        attentions = (attentions[0][0], *attentions[1:])
        attentions = [attn.data.cpu() for attn in attentions]
        attn_rollout = attention_rollout_base_transformer_encoder(attentions, rollout_discard_ratio)
        
        attn_vectors = [x[0] for x in attn_rollout]
        attn_vector = attn_vectors[0]
        norm_attn_vector = (attn_vector - torch.min(attn_vector)) / (torch.max(attn_vector) - torch.min(attn_vector))
        heatmap = np.zeros(shape=reduced_size)
        for attn, idx in zip(norm_attn_vector, idcs):
            x, y = idx
            heatmap[(x, y)] = attn

        base_fn = os.path.split(fn)[1]
        new_files_basename = '.'.join(base_fn.split('.')[:-1])
        heatmap_fn = new_files_basename + '_heatmap.png'
        array_fn = new_files_basename + '_heatmap_array.pkl'
        
        with open(array_fn, 'wb') as h:
            pickle.dump(heatmap, h)

        plt.imsave(heatmap_fn, heatmap, format='png', cmap='plasma')

if __name__ == '__main__':
    task()
