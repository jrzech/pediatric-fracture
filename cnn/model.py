from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# image imports
from skimage import io, transform
from PIL import Image
import cv2

# general imports
import os
import time
from shutil import copyfile, rmtree

# data science imports
import pandas as pd
import numpy as np
import csv

import rad_dataset as D
import eval_model as E
import torchvision.transforms.functional as F

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))


def checkpoint(model, best_loss, epoch, LR,optimizer,scheduler):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR,
        'optimizer':optimizer,
        'scheduler':scheduler
    }

    torch.save(state, 'results/checkpoint')
    
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,PATH_TO_IMAGES,data_transforms):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'tune']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                right = 0

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, names = data
                check=inputs.cpu().data.numpy()
                #this lets one see example images as they are preprocessed and presented to model, can be useful for debugging preprocessing mistakes
                DEBUG_IMAGE_PROCESSING=False
                if DEBUG_IMAGE_PROCESSING:
                    for k in range(0,inputs.shape[0]):
                        print(check[k].shape)
                        thisarray=np.moveaxis(check[k], 0, -1)
                        thisarray=rgb2gray(thisarray)
                        thisarray=thisarray-thisarray.min()
                        thisarray=thisarray/thisarray.max()
                        thisarray=thisarray*255
                        thisarray=np.where(thisarray>255,255,thisarray)
                        thisarray=np.where(thisarray<0,0,thisarray)
                        thisarray=thisarray.astype(np.uint8)
                        thisarray=np.where(thisarray>255,255,thisarray)
                        thisarray=np.where(thisarray<0,0,thisarray)
                        print(thisarray.max())
                        print(thisarray.min())
                        print(thisarray.shape)
                        Image.fromarray(thisarray).save("/home/jrzech/research/testimg/"+"label_"+str(labels.cpu().data.numpy()[k])+"_"+str(names[k]).replace("/","_"))
                        print(names[k])
                        print(labels[k])
                
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float() 

                optimizer.zero_grad()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                if phase== 'tune':
                    probs = outputs.cpu().data.numpy()
                    truth = labels.cpu().data.numpy()
                    for i in range(0,probs.shape[0]):
                        if probs[i]>=0.5 and truth[i] == 1: right+=1
                        if probs[i]<0.5 and truth[i] == 0: right+=1
                

                running_loss += loss.item() * batch_size #was loss.data

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase=='tune':
                #print("scheduler.step()")
                scheduler.step(epoch_loss)
                #print(optimizer) 
            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))
            if phase == 'tune': print("tune acc: "+str(right/dataset_sizes[phase]))

            #take data off gpu
            inputs.cpu()
            labels.cpu()
            
            # checkpoint model if has best val loss yet
            if phase == 'tune' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR, optimizer, scheduler)
                #E.make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES)

            # log training and validation loss over each epoch
            if phase == 'tune':
                with open("results/log_train", 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "tune_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in X epochs
        if ((epoch - best_epoch) >= 5):
            print("no improvement in 5 epochs, break")
            break
        end = time.time()
        print("epoch time "+str((end-start)/60)+ " minutes")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch


def train_cnn(PATH_TO_IMAGES, LR, WEIGHT_DECAY,IMAGE_SIZE,AUGMENT,CLAHE,BATCH_SIZE,USE_METADATA,BODYPARTS,SAMPLE,OPTIMIZER,TARGET,EPOCHS):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """
    NUM_EPOCHS = EPOCHS
    BATCH_SIZE = BATCH_SIZE

    print("USE METADATA is")
    print(USE_METADATA)

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")

    # use imagenet mean,std for normalization by convention
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = len(TARGET)  # we are predicting N labels

    # define torchvision transforms
    SIZE=IMAGE_SIZE
    
    if (AUGMENT and USE_METADATA):
        print("note!! USE_METADATA so suppress randombrightnesscontrast augmentation")
        data_transforms = { 
            'train':A.Compose([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=25, p=0.5, border_mode=cv2.BORDER_CONSTANT,value=0,interpolation=cv2.INTER_AREA, crop_border=True),            
                A.LongestMaxSize(max_size=SIZE+int(SIZE/4), interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=SIZE+int(SIZE/4), min_width=SIZE+int(SIZE/4), border_mode=0, value=(0,0,0)),
                A.RandomSizedCrop(min_max_height=[SIZE,SIZE],height=SIZE, width=SIZE, w2h_ratio=1,p=0.5),
                #A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_AREA, p=1),
                ToTensorV2(p=1.0),
            ]),
            'tune': A.Compose([
                A.LongestMaxSize(max_size=SIZE, interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=SIZE, min_width=SIZE, border_mode=0, value=(0,0,0)),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_AREA, p=1.0),
                ToTensorV2(p=1.0),
            ]),
        }
    elif AUGMENT:
        print("use REGULAR augmentations")
        data_transforms = {
            'train':A.Compose([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=25, p=0.5, border_mode=cv2.BORDER_CONSTANT,value=0,interpolation=cv2.INTER_AREA, crop_border=True),
                A.LongestMaxSize(max_size=SIZE+int(SIZE/4), interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=SIZE+int(SIZE/4), min_width=SIZE+int(SIZE/4), border_mode=0, value=(0,0,0)),
                A.RandomSizedCrop(min_max_height=[SIZE,SIZE],height=SIZE, width=SIZE, w2h_ratio=1,p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_AREA, p=1),
                ToTensorV2(p=1.0),
            ]),
            'tune': A.Compose([
                A.LongestMaxSize(max_size=SIZE, interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=SIZE, min_width=SIZE, border_mode=0, value=(0,0,0)),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_AREA, p=1.0),
                ToTensorV2(p=1.0),
            ]),
        }

    else:
        print("no augmentation!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_transforms = {
            'train': A.Compose([
                A.LongestMaxSize(max_size=SIZE, interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=SIZE, min_width=SIZE, border_mode=0, value=(0,0,0)),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_AREA, p=1.0),
                ToTensorV2(p=1.0),
            ]),
            'tune': A.Compose([
                A.LongestMaxSize(max_size=SIZE, interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=SIZE, min_width=SIZE, border_mode=0, value=(0,0,0)),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_AREA, p=1.0),
                ToTensorV2(p=1.0),
            ])
        }


    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = D.RadDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'],
        sample=SAMPLE,
        clahe=CLAHE,usemetadata=USE_METADATA,bodyparts=BODYPARTS,target=TARGET)
    transformed_datasets['tune'] = D.RadDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='tune',
        transform=data_transforms['tune'],
        sample=SAMPLE,
        clahe=CLAHE,usemetadata=USE_METADATA,bodyparts=BODYPARTS,target=TARGET)

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0)
    dataloaders['tune'] = torch.utils.data.DataLoader(
        transformed_datasets['tune'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0)

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    
    #torchvision efficientnet. so many others one can try in torchvision - efficientnet_v2_l, efficientnet_b7, etc.
    model = torchvision.models.efficientnet_v2_s(weights='DEFAULT')
    #specifics of how one modifies imagenet-pretrained model to your # classes depends on model but
    #effnet implementations follow this convention:
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())    

    #for example if we wanted to use other models:
    #model = torchvision.models.ViT_L_16(ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    #num_ftrs = model.heads[0].in_features #for ViT

    #print("LOAD PRETRAIN")
    #checkpoint_best = torch.load('/home/jrzech/research/code/cnn/results-pedue-cropnorm-8bit-pickup/checkpoint')
    #model = checkpoint_best['model'] 
    
    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    
    if OPTIMIZER=="SGD":
        optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9,weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER=="Adam":
        optimizer = optim.Adam(model.parameters(),lr=LR, betas=(0.9, 0.999),weight_decay=WEIGHT_DECAY)
    else:
        print("undefined optimizer: "+print(OPTIMIZER))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=(1/np.sqrt(10)),patience=2) 

    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'tune']}

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, scheduler, LR, num_epochs=NUM_EPOCHS, dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY,PATH_TO_IMAGES=PATH_TO_IMAGES,data_transforms=data_transforms)

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms, model, PATH_TO_IMAGES,IMAGE_SIZE,CLAHE,USE_METADATA,BODYPARTS)

    return preds, aucs
