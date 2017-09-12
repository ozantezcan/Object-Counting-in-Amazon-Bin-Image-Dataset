import torch
from torchvision import datasets
import os

def load_data(dataset, data_transforms, uniform_sampler=True, batch_size=16):
    if dataset == 'real':
        data_dir = '..//Data_Sets//pruned//good'
        dsets = {x: datasets.ImageFolder_mtezcan([os.path.join(data_dir, x)], data_transforms[x])
                 for x in ['train', 'val']}
        dsets_real = dsets

    if dataset == 'mit_indoor':
        data_dir = '..//Data_Sets//MIT_indoor_scenes'
        dsets = {x: datasets.ImageFolder_mtezcan([os.path.join(data_dir, x)], data_transforms[x])
                 for x in ['train', 'val']}
        dsets_real = dsets

    if dataset == 'synthetic':
        rootdir = '//media//mtezcan//New Volume/HoardingImages//_rated//'
        valdir = '//media//mtezcan//New Volume/HoardingImages//_val//validation//House//BR//'
        # subdirs=os.listdir(rootdir)
        subdirs = ['BasicHouse_2017-07-01-rated',
                   'BriansHouse_2017-06-30-rated',
                   'RuralHome_2017-06-30-rated',
                   'SmallApt_2017-06-29-rated']
        roomdirs = ['//BR', '//Kitchen', '//LR']
        '''
        rootdir='//media//mtezcan//New Volume/HoardingImages//_rated//train//'
        valdir='//media//mtezcan//New Volume/HoardingImages//_rated//val//BriansHouse_2017-06-30-rated//LR//'
        subdirs=['BriansHouse_2017-06-30-rated']
        print(subdirs)
        roomdirs=['//LR']
        '''
        dsets = {'train': datasets.ImageFolder_mtezcan([rootdir + subdir + room for subdir in subdirs
                                                        for room in roomdirs], data_transforms['train']),
                 'val': datasets.ImageFolder_mtezcan([valdir], data_transforms['val'])}

    if uniform_sampler:
        weights, wpc = ft.make_weights_for_balanced_classes(dsets['train'].imgs, len(dsets['train'].classes))
        weights = torch.DoubleTensor(weights)
        sampler = {'train': torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights)),
                   'val': None}
    else:
        sampler = {'train': None,
                   'val': None}

    shuffler = {'train': True, 'val': False}
    dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x], batch_size=batch_size, shuffle=shuffler[x], sampler=sampler[x], num_workers=12)
    for x in ['train', 'val']}
    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes
    print(dset_classes)
    use_gpu = torch.cuda.is_available()

    if (dataset == 'real'):
        dset_loaders_real = dset_loaders
        dset_sizes_real = dset_sizes
    elif (dataset == 'synthetic'):
        dset_loaders_synthetic = dset_loaders
        dset_sizes_synthetic = dset_sizes
     # use_gpu=False
    if use_gpu:
        print('GPU is available')
    else:
        print('!!!!! NO CUDA GPUS DETECTED')