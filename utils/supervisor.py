import  torch
from torch import  nn
import  torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import argparse
import config


"""
In our defensive setting, we assume a poisoned training set and a small clean 
set at hand, i.e. we train base model jointly with the poisoned set and 
the shifted set (constructed based on the small clean set).

On the other hand, we also prepare a clean test set (usually larger than the 
small clean set used for defense in our experiments). But note that, this set is 
not used for defense, it is used for debug and evaluation!

Below we implement tools to take use of the additional clean test set for debug & evaluation.
"""

def get_cleansed_set_indices_dir(args):
    poison_set_dir = get_poison_set_dir(args)
    if args.cleanser == 'CT': # confusion training    
        return os.path.join(poison_set_dir, f'cleansed_set_indices_seed={args.seed}')
    else:
        return os.path.join(poison_set_dir, f'cleansed_set_indices_other_{args.cleanser}_seed={args.seed}')

def get_model_name(args, cleanse=False):
    # `args.model_path` > `args.model` > by default 'full_base'
    if hasattr(args, 'model_path') and args.model_path is not None:
        model_name = args.model_path
    elif hasattr(args, 'model') and args.model is not None:
        model_name = args.model
    else:
        if args.no_aug:
            model_name = f'full_base_no_aug_seed={args.seed}.pt'
        else:
            model_name = f'full_base_aug_seed={args.seed}.pt'
        
        if cleanse and hasattr(args, 'cleanser') and args.cleanser is not None:
            model_name = f"cleansed_{args.cleanser}_{model_name}"
    return model_name

def get_model_dir(args, cleanse=False):
    if hasattr(args, 'model_path') and args.model_path is not None:
        return args.model_path
    else:
        return f"{get_poison_set_dir(args)}/{get_model_name(args, cleanse=cleanse)}"

def get_dir_core(args, include_model_name=False, include_poison_seed=False):
    ratio = '%.3f' % args.poison_rate
    # ratio = '%.1f' % (args.poison_rate * 100) + '%'
    if args.poison_type == 'blend':
        blend_alpha = '%.3f' % args.alpha
        dir_core = '%s_%s_%s_alpha=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, args.trigger)
    elif args.poison_type == 'adaptive_blend':
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        dir_core = '%s_%s_%s_alpha=%s_cover=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger)
    elif args.poison_type == 'adaptive_k_way' or args.poison_type == 'adaptive_patch' or args.poison_type == 'TaCT':
        cover_rate = '%.3f' % args.cover_rate
        dir_core = '%s_%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    else:
        dir_core = '%s_%s_%s' % (args.dataset, args.poison_type, ratio)

    if include_model_name:
        dir_core = f'{dir_core}_{get_model_name(args)}'
    if include_poison_seed:
        dir_core = f'{dir_core}_poison_seed={config.poison_seed}'
    if config.record_model_arch:
        dir_core = f'{dir_core}_arch={config.arch[args.dataset].__name__}'
    return dir_core

def get_poison_set_dir(args):
    ratio = '%.3f' % args.poison_rate
    # ratio = '%.1f' % (args.poison_rate * 100) + '%'
    if args.poison_type == 'blend':
        blend_alpha = '%.3f' % args.alpha
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, args.trigger)
    elif args.poison_type == 'adaptive_blend':
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_cover=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger)
    elif args.poison_type == 'adaptive_k_way' or args.poison_type == 'adaptive_patch' or args.poison_type == 'TaCT':
        cover_rate = '%.3f' % args.cover_rate
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    else:
        poison_set_dir = 'poisoned_train_set/%s/%s_%s' % (args.dataset, args.poison_type, ratio)
    
    if config.record_poison_seed: poison_set_dir = f'{poison_set_dir}_poison_seed={config.poison_seed}' # debug
    if config.record_model_arch: poison_set_dir = f'{poison_set_dir}_arch={config.arch[args.dataset].__name__}'
    return poison_set_dir

def get_poison_transform(poison_type, dataset_name, target_class, source_class=1, cover_classes=[5,7],
                         is_normalized_input = False, trigger_transform=None,
                         alpha=0.2, trigger_name=None, args = None):

    # source class will be used for TaCT poison

    if trigger_name is None:
        trigger_name = config.trigger_default[poison_type]

    if dataset_name in ['gtsrb','cifar10', 'cifar100']:
        img_size = 32
    elif dataset_name == 'imagenette':
        img_size = 224
    else:
        raise NotImplementedError('<Undefined> Dataset = %s' % dataset_name)

    if dataset_name == 'cifar10':
        normalizer = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                    [1 / 0.247, 1 / 0.243, 1 / 0.261])
        ])
    elif dataset_name == 'gtsrb':
        normalizer = transforms.Compose([
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
                                    (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
        ])
    else:
        raise Exception("Invalid Dataset")

    poison_transform = None
    trigger = None
    trigger_mask = None

    if poison_type in ['none', 'badnet', 'blend',
                       'adaptive_blend', 'adaptive_patch', 'adaptive_k_way']:

        if trigger_transform is None:
            trigger_transform = transforms.Compose([
                transforms.ToTensor()
            ])

        if trigger_name != 'none': # none for SIG
            trigger_path = os.path.join(config.triggers_dir, trigger_name)
            trigger = Image.open(trigger_path).convert("RGB")

            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_name)

            if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
            else:  # by default, all black pixels are masked with 0's
                temp_trans = transforms.ToTensor()
                trigger_map = temp_trans(trigger)
                trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0), trigger_map[2] > 0).float()

            trigger = trigger_transform(trigger).cuda()
            trigger_mask = trigger_mask.cuda()

        if poison_type == 'badnet':
            from poison_tool_box import badnet
            poison_transform = badnet.poison_transform(img_size=img_size, trigger=trigger, target_class=target_class)

        elif poison_type == 'blend':
            from poison_tool_box import blend
            poison_transform = blend.poison_transform(img_size=img_size, trigger=trigger,
                                                      target_class=target_class, alpha=alpha)
        
        elif poison_type == 'adaptive_blend':
            from poison_tool_box import adaptive_blend
            poison_transform = adaptive_blend.poison_transform(img_size=img_size, trigger=trigger,
                                                               target_class=target_class, alpha=alpha)
        
        elif poison_type == 'adaptive_patch':
            from poison_tool_box import adaptive_patch
            poison_transform = adaptive_patch.poison_transform(img_size=img_size, test_trigger_names=config.adaptive_patch_test_trigger_names[args.dataset], test_alphas=config.adaptive_patch_test_trigger_alphas[args.dataset], target_class=target_class, denormalizer=denormalizer, normalizer=normalizer,)

        elif poison_type == 'adaptive_k_way':
            from poison_tool_box import adaptive_k_way
            poison_transform = adaptive_k_way.poison_transform(img_size=img_size, target_class=target_class, denormalizer=denormalizer, normalizer=normalizer,)

        else: # 'none'
            from poison_tool_box import none
            poison_transform = none.poison_transform()

        return poison_transform


    else:
        raise NotImplementedError('<Undefined> Poison_Type = %s' % poison_type)