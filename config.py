from utils import resnet, wresnet, vgg, mobilenetv2
from utils import supervisor
from utils import tools
import torch
from torchvision import transforms
import os


data_dir = './data' # defaul clean dataset directory
triggers_dir = './triggers' # default triggers directory
target_class = {
    'cifar10' : 0,
    'gtsrb' : 2
}

# default target class (without loss of generality)
source_class = 1 # default source class for TaCT
cover_classes = [5,7] # default cover classes for TaCT
seed = 666 # 999, 999, 666 (1234, 5555, 777)
poison_seed = 0
record_poison_seed = False
record_model_arch = False

parser_choices = {
    'dataset': ['gtsrb','cifar10', 'cifar100', 'imagenette'],
    'poison_type': ['badnet', 'blend', 'adaptive_blend', 'adaptive_patch', 'adaptive_k_way', 'none'],
    'poison_rate': [i / 1000.0 for i in range(0, 500)],
    'cover_rate': [i / 1000.0 for i in range(0, 500)],
}

parser_default = {
    'dataset': 'cifar10',
    'poison_type': 'badnet',
    'poison_rate': 0,
    'cover_rate': 0,
    'alpha': 0.2,
}

trigger_default = {
    'adaptive_blend': 'hellokitty_32.png',
    'adaptive_k_way': 'none',
    'adaptive_patch': 'none',
    'badnet' : 'badnet_patch.png',
    'blend' : 'hellokitty_32.png',
    'none' : 'none',
}

arch = {
    'cifar10': resnet.resnet20,
    # 'cifar10': vgg.vgg16_bn,
    # 'cifar10': mobilenetv2.mobilenetv2,
    'gtsrb' : resnet.resnet20,
    'abl':  wresnet.WideResNet
}

# adapitve-patch triggers for different datasets
adaptive_patch_train_trigger_names = {
    'cifar10': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
    'gtsrb': [
        'phoenix_corner_32.png',
        'firefox_corner_32.png',
        'badnet_patch4_32.png',
        'trojan_square_32.png',
    ],
}

adaptive_patch_train_trigger_alphas = {
    'cifar10': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
    'gtsrb': [
        0.5,
        0.2,
        0.5,
        0.3,
    ],
}

adaptive_patch_test_trigger_names = {
    'cifar10': [
        'phoenix_corner2_32.png',
        'badnet_patch4_32.png',
    ],
    'gtsrb': [
        'firefox_corner_32.png',
        'trojan_square_32.png',
    ],
}

adaptive_patch_test_trigger_alphas = {
    'cifar10': [
        1,
        1,
    ],
    'gtsrb': [
        1,
        1,
    ],
}


def get_params(args):


    if args.dataset == 'cifar10':

        num_classes = 10

        data_transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])

        data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ])

        lamb1 = 24.0
        lamb2 = 24.0

        lr_base = 0.1
        lr_distillation = 0.01
        lr_inference = 0.01

        condensation_num = 2000
        median_sample_rate = 0.1
        weight_decay = 1e-4

    elif args.dataset == 'gtsrb':

        num_classes = 43

        data_transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

        data_transform_aug = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

        # batch_id // 10 , with lamb = 4.0 => good ..
        lamb1 = 14.0
        lamb2 = 14.0

        lr_base = 0.1
        lr_distillation = 0.001
        lr_inference = 0.001

        condensation_num = 2000
        median_sample_rate = 0.1
        weight_decay = 1e-4

    else:
        raise NotImplementedError('<Unimplemented Dataset> %s' % args.dataset)

    from utils import wide_resnet
    params ={
        'data_transform' : data_transform_normalize,
        'data_transform_aug' : data_transform_aug,
        'lamb_distillation' : lamb1,
        'lamb_inference' : lamb2,
        'lr_base' : lr_base,
        'lr_distillation' : lr_distillation,
        'lr_inference' : lr_inference,
        'weight_decay' : weight_decay,
        'condensation_num' : condensation_num, # number of samples extracted after distillation (samples with least losses will be extracted)
        'median_sample_rate' : median_sample_rate, # rate of samples extracted from the sorted samples to approximate the clean statistics
        'distillation_ratio' : [1/2, 1/4],

        'num_classes' : num_classes,
        'batch_size' : 128,
        'pretrain_epochs' : 60,
        'base_arch' :  arch[args.dataset],
        'inference_arch' :  arch['low_dim'],

        'inspection_set_dir' : supervisor.get_poison_set_dir(args)
    }

    return params


def get_dataset(inspection_set_dir, data_transform, args):

    # Set Up Inspection Set (dataset that is to be inspected
    inspection_set_img_dir = os.path.join(inspection_set_dir, 'data')
    inspection_set_label_path = os.path.join(inspection_set_dir, 'labels')
    inspection_set = tools.IMG_Dataset(data_dir=inspection_set_img_dir,
                                     label_path=inspection_set_label_path, transforms=data_transform)

    # Set Up Clean Set (the small clean split at hand for defense
    clean_set_dir = os.path.join('clean_set', args.dataset, 'clean_split')
    clean_set_img_dir = os.path.join(clean_set_dir, 'data')
    clean_label_path = os.path.join(clean_set_dir, 'clean_labels')
    clean_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                  label_path=clean_label_path, transforms=data_transform)

    return inspection_set, clean_set


def get_packet_for_debug(poison_set_dir, data_transform, batch_size, args):

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)


    kwargs = {'num_workers': 2, 'pin_memory': True}
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=True, **kwargs)

    trigger_transform = data_transform
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=target_class[args.dataset],
                                                       trigger_transform=trigger_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)

    poison_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))

    if args.poison_type == 'TaCT':
        source_classes = [source_class]
    else:
        source_classes = None

    debug_packet = {
        'test_set_loader' : test_set_loader,
        'poison_transform' : poison_transform,
        'poison_indices' : poison_indices,
        'source_classes' : source_classes
    }

    return debug_packet
