import os
import torch
from torchvision import datasets, transforms
import argparse
from PIL import Image
import numpy as np
import config
from utils import supervisor, tools

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, required=False,
                    default=config.parser_default['dataset'],
                    choices=config.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=config.parser_choices['poison_type'],
                    default=config.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=config.parser_choices['poison_rate'],
                    default=config.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=config.parser_choices['cover_rate'],
                    default=config.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False,
                    default=config.parser_default['alpha'])
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
args = parser.parse_args()

tools.setup_seed(0)


print('[target class : %d]' % config.target_class[args.dataset])

data_dir = config.data_dir  # directory to save standard clean set
if args.trigger is None:
    args.trigger = config.trigger_default[args.poison_type]

if not os.path.exists(os.path.join('poisoned_train_set', args.dataset)):
    os.mkdir(os.path.join('poisoned_train_set', args.dataset))

if args.dataset == 'gtsrb':
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split = 'train',
                                transform = data_transform, download=True)
    img_size = 32
    num_classes = 43
elif args.dataset == 'cifar10':
    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                    download=True, transform=data_transform)
    img_size = 32
    num_classes = 10
elif args.dataset == 'cifar100':
    raise NotImplementedError('cifar100 unsupported!')
elif args.dataset == 'imagenette':
    raise  NotImplementedError('imagenette unsupported!')
else:
    raise  NotImplementedError('Undefined Dataset')

trigger_transform = transforms.Compose([
    transforms.ToTensor()
])

# Create poisoned dataset directory for current setting
poison_set_dir = supervisor.get_poison_set_dir(args)
poison_set_img_dir = os.path.join(poison_set_dir, 'data')

if not os.path.exists(poison_set_dir):
    os.mkdir(poison_set_dir)
if not os.path.exists(poison_set_img_dir):
    os.mkdir(poison_set_img_dir)



if args.poison_type in ['badnet', 'blend', 'none',
                        'adaptive_blend', 'adaptive_patch', 'adaptive_k_way']:

    trigger_name = args.trigger
    trigger_path = os.path.join(config.triggers_dir, trigger_name)

    trigger = None
    trigger_mask = None

    if trigger_name != 'none':  # none for SIG

        print('trigger: %s' % trigger_path)

        trigger_path = os.path.join(config.triggers_dir, trigger_name)
        trigger = Image.open(trigger_path).convert("RGB")
        trigger = trigger_transform(trigger)

        trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_name)
        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            #print('trigger_mask_path:', trigger_mask_path)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            #print('No trigger mask found! By default masking all black pixels...')
            trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0), trigger[2] > 0).float()

    alpha = args.alpha

    poison_generator = None

    if args.poison_type == 'badnet':

        from poison_tool_box import badnet
        poison_generator = badnet.poison_generator(img_size=img_size, dataset=train_set,
                                                   poison_rate=args.poison_rate, trigger=trigger,
                                                   path=poison_set_img_dir, target_class=config.target_class[args.dataset])

    elif args.poison_type == 'blend':

        from poison_tool_box import blend
        poison_generator = blend.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate, trigger=trigger,
                                                  path=poison_set_img_dir, target_class=config.target_class[args.dataset],
                                                  alpha=alpha)

    elif args.poison_type == 'adaptive_blend':

        from poison_tool_box import adaptive_blend
        poison_generator = adaptive_blend.poison_generator(img_size=img_size, dataset=train_set,
                                                          poison_rate=args.poison_rate,
                                                          path=poison_set_img_dir, trigger=trigger,
                                                          pieces=16, mask_rate=0.5,
                                                          target_class=config.target_class[args.dataset], alpha=alpha,
                                                          cover_rate=args.cover_rate)
    
    elif args.poison_type == 'adaptive_k_way':

        from poison_tool_box import adaptive_k_way
        poison_generator = adaptive_k_way.poison_generator(img_size=img_size, dataset=train_set,
                                                           poison_rate=args.poison_rate,
                                                           path=poison_set_img_dir,
                                                           target_class=config.target_class[args.dataset],
                                                           cover_rate=args.cover_rate)
    
    elif args.poison_type == 'adaptive_patch':


        from poison_tool_box import adaptive_patch
        poison_generator = adaptive_patch.poison_generator(img_size=img_size, dataset=train_set,
                                                           poison_rate=args.poison_rate,
                                                           path=poison_set_img_dir,
                                                           trigger_names=config.adaptive_patch_train_trigger_names[args.dataset],
                                                           alphas=config.adaptive_patch_train_trigger_alphas[args.dataset],
                                                           target_class=config.target_class[args.dataset],
                                                           cover_rate=args.cover_rate)

    else: # 'none'
        from poison_tool_box import none
        poison_generator = none.poison_generator(img_size=img_size, dataset=train_set,
                                                path=poison_set_img_dir)



    if args.poison_type not in ['adaptive_blend', 'adaptive_patch', 'adaptive_k_way']:
        poison_indices, label_set = poison_generator.generate_poisoned_training_set()
        print('[Generate Poisoned Set] Save %d Images' % len(label_set))

    else:
        poison_indices, cover_indices, label_set = poison_generator.generate_poisoned_training_set()
        print('[Generate Poisoned Set] Save %d Images' % len(label_set))

        cover_indices_path = os.path.join(poison_set_dir, 'cover_indices')
        torch.save(cover_indices, cover_indices_path)
        print('[Generate Poisoned Set] Save %s' % cover_indices_path)



    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

    #print('poison_indices : ', poison_indices)

else:
    raise NotImplementedError('%s not defined' % args.poison_type)