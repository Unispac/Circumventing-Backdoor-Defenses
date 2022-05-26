#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import argparse
import config
from utils import supervisor
from . import BackdoorDefense
from .tools import to_list, generate_dataloader, val_atk


class FP(BackdoorDefense):
    """
    Fine Pruning Defense is described in the paper 'Fine-Pruning'_ by KangLiu. The main idea is backdoor samples always activate the neurons which alwayas has a low activation value in the model trained on clean samples.

    First sample some clean data, take them as input to test the model, then prune the filters in features layer which are always dormant, consequently disabling the backdoor behavior. Finally, finetune the model to eliminate the threat of backdoor attack.

    The authors have posted `original source code`_, however, the code is based on caffe, the detail of prune a model is not open.

    Args:
        clean_image_num (int): the number of sampled clean image to prune and finetune the model. Default: 50.
        prune_ratio (float): the ratio of neurons to prune. Default: 0.02.
        # finetune_epoch (int): the epoch of finetuning. Default: 10.


    .. _Fine Pruning:
        https://arxiv.org/pdf/1805.12185


    .. _original source code:
        https://github.com/kangliucn/Fine-pruning-defense

    .. _related code:
        https://github.com/jacobgil/pytorch-pruning
        https://github.com/eeric/channel_prune


    """

    def __init__(self, args, prune_ratio: float = 0.95, finetune_epoch=10, max_allowed_acc_drop=0.2):
        super().__init__(args)
        
        self.args = args
        self.prune_ratio = prune_ratio
        self.finetune_epoch = finetune_epoch
        self.max_allowed_acc_drop = max_allowed_acc_drop

        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                last_conv = module
                self.prune_layer: str = name
                break
        else:
            raise Exception('There is no Conv2d in model.')
        length = last_conv.out_channels
        self.prune_num = int(length * self.prune_ratio)
        self.folder_path = 'other_defenses/results/FP'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        self.valid_loader = generate_dataloader(dataset=self.dataset,
                                    dataset_path=config.data_dir,
                                    batch_size=100,
                                    split='valid',
                                    shuffle=True,
                                    drop_last=False)

    def detect(self):
        self.ori_clean_acc = val_atk(self.args, self.model)[0]
        self.prune()

    def prune(self):
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                self.last_conv: nn.Conv2d = prune.identity(module, 'weight')
                break
        length = self.last_conv.out_channels

        mask: torch.Tensor = self.last_conv.weight_mask
        
        assert self.prune_num >= self.finetune_epoch, "prune_ratio too small!"
        self.prune_step(mask, prune_num=max(self.prune_num - self.finetune_epoch, 0))
        val_atk(self.args, self.model)

        for i in range(min(self.finetune_epoch, length)):
            print('Iter: %d/%d' % (i + 1, min(self.finetune_epoch, length)))
            self.prune_step(mask, prune_num=1)
            clean_acc = val_atk(self.args, self.model)[0]
            if self.ori_clean_acc - clean_acc > self.max_allowed_acc_drop: # stop if accuracy drop too much
                break
        
        # test pruned model and save
        result_file = os.path.join(self.folder_path, 'FP_%s.pt' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
        torch.save(self.model.state_dict(), result_file)
        print('Fine-Pruned Model Saved at:', result_file)
        val_atk(self.args, self.model)

    @torch.no_grad()
    def prune_step(self, mask: torch.Tensor, prune_num: int = 1):
        if prune_num <= 0: return
        feats_list = []
        for _input, _label in self.valid_loader:
            _input, _label = _input.cuda(), _label.cuda()
            _, _feats = self.model.forward(_input, return_hidden=True)
            _feats = _feats.abs()
            # if _feats.dim() > 2:
            #     _feats = _feats.flatten(2).mean(2)
            feats_list.append(_feats)
        feats_list = torch.cat(feats_list).mean(dim=0)
        idx_rank = feats_list.argsort()
        counter = 0
        for idx in idx_rank:
            if mask[idx].norm(p=1) > 1e-6:
                mask[idx] = 0.0
                counter += 1
                print(f'[{counter}/{prune_num}] Pruned channel id {idx}/{len(idx_rank)}')
                if counter >= min(prune_num, len(idx_rank)):
                    break