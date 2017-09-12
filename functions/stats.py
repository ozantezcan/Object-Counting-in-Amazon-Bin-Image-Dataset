from __future__ import print_function, division

import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

default_transform=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def im2torchNorm(imdir,mean = np.array([0.485, 0.456, 0.406]),std = np.array([0.229, 0.224, 0.225])\
                 ,imsize=(256,256),imMax=255.):
    im = Image.open(imdir)
    im = im.resize(imsize)#, Image.ANTIALIAS) # resizes image in-place
    im=np.asarray(im).astype(np.float)/imMax
    im=im[16:240,16:240,:]
    im_norm=(im-mean)/std
    return im_norm


def im2torchTransform(imdir, transform=default_transform):
    im = Image.open(imdir)
    return transform(im).numpy().transpose(1,2,0)

def subsetCreator(rootdir,im_per_room=10,roomdirs=['//BR//','//Kitchen//','//LR//'],multi_dir=True):
    if(multi_dir):
    	subdirs=os.listdir(rootdir)
    else:
        subdirs=['']
    imdirs=[]
    cir=[]
    room=[]
    house=[]

    for hme in range(len(subdirs)):
        for rm in range(len(roomdirs)):
            for cr in range(9):
                parentdir=rootdir+subdirs[hme]+roomdirs[rm]+str(cr+1)
                if os.path.exists(parentdir):
                    imdirs_c=os.listdir(parentdir)
                    if(len(imdirs_c)>0):
                        if im_per_room==0:
                            rand_idx=range(len(imdirs_c))
                        else:
                            rand_idx=np.floor(np.random.rand(im_per_room)*len(imdirs_c)).astype(np.int)
                        for idx in rand_idx:
                            imdirs.append(parentdir+'//'+imdirs_c[idx])
                            house.append(hme+1)
                            room.append(rm+1)
                            cir.append(cr+1)
    return np.asarray(imdirs), np.asarray(cir), np.asarray(house), np.asarray(room)



def torchFromDirs(imdirs,im_dims=[224,224,3],begin_idx=0,batch_size=16):
    if len(imdirs)<begin_idx:
        raise ValueError('Begin index cannot be higher than the length of image dirs')
    if len(imdirs)<begin_idx+batch_size:
        batch_size=len(imdirs)-begin_idx
    imgs=np.zeros([batch_size]+im_dims)
    for k in range(batch_size):
        imgs[k,:,:,:]=im2torchTransform(imdirs[begin_idx+k])

    im_torch=Variable(torch.from_numpy(imgs.transpose(0,3,1,2)).cuda()).float()
    return im_torch

def class_based_cirs(label,pred):
    cirs=np.zeros(10)
    cir1s=np.zeros(10)
    cir2s=np.zeros(10)

    for k in range(9):
        cir=k+1
        idxs= label==cir
        label_cir=label[idxs]
        pred_cir=pred[idxs]
        err=np.abs(label_cir-pred_cir)
        cirs[k]=np.mean(err<=0)
        cir1s[k]=np.mean(err<=1)
        cir2s[k]=np.mean(err<=2)

    err=np.abs(label-pred)
    cirs[9]=np.mean(err<=0)
    cir1s[9]=np.mean(err<=1)
    cir2s[9]=np.mean(err<=2)
    return cirs,cir1s,cir2s

def extractFeats(imdirs,network,batchsize=16,outsize=512):
    fvec=np.zeros([len(imdirs),outsize])

    for k in range(0,len(imdirs),batchsize):
        im_torch=torchFromDirs(imdirs,begin_idx=k,batch_size=batchsize)
        feat=network(im_torch)
        feat=feat.cpu()
        feat=feat.data.numpy()
        fvec[k:k+batchsize]=feat.reshape(-1,outsize)
    return fvec

def plotAll(fvec_tsne,cir,house,room,data_title=''):
    plt.figure()
    plt.scatter(fvec_tsne[:,0], fvec_tsne[:,1], c=cir, alpha=0.5)#,cmap='jet')
    plt.colorbar()
    plt.title(data_title+'Images by CIR')

    plt.figure()
    plt.scatter(fvec_tsne[:,0], fvec_tsne[:,1], c=house, alpha=0.5)#,cmap='jet')
    plt.colorbar()
    plt.title(data_title+'Images by House')

    plt.figure()
    plt.scatter(fvec_tsne[:,0], fvec_tsne[:,1], c=room, alpha=0.5)#,cmap='jet')
    plt.colorbar()
    plt.title(data_title+'Images by Room')

