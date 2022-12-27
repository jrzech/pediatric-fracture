import torch
import pandas as pd
import rad_dataset as D
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn
import sklearn.metrics as sklm
from torch.autograd import Variable
import numpy as np
import warnings

#don't want to see deprecation warnings for pd append rn
warnings.filterwarnings("ignore")


def make_pred_multilabel(data_transforms, model, PATH_TO_IMAGES,IMAGE_SIZE,CLAHE,USE_METADATA,BODYPARTS):
    """
    Gives predictions for test fold and calculates AUCs using previously trained model

    Args:
        data_transforms: albumentations transforms to preprocess raw images; same as validation transforms
        model: effnetv2s from torchvision previously fine tuned to training data
        PATH_TO_IMAGES: path at which images can be found
    Returns:
        pred_df: dataframe containing individual predictions and ground truth for each test image
        auc_df: dataframe containing aggregate AUCs by train/test tuples
    """

    # calc preds in batches of 16, can reduce if your GPU has less RAM
    BATCH_SIZE = 1

    # set model to eval mode; required for proper predictions given use of batchnorm
    model.train(False)

    # create dataloader
    dataset = D.RadDataset(
        path_to_images=PATH_TO_IMAGES,
        fold="test",
        transform=data_transforms['tune'],clahe=CLAHE,usemetadata=USE_METADATA,bodyparts=BODYPARTS)
    dataloader = torch.utils.data.DataLoader(
        dataset, BATCH_SIZE, shuffle=False, num_workers=0)
    size = len(dataset)

    # create empty dfs
    FILENAME="filename"
    FINDING ="fx"
    pred_df = pd.DataFrame()#columns=[FILENAME])
    true_df = pd.DataFrame()#columns=[FILENAME])

    # iterate over dataloader
    for i, data in enumerate(dataloader):

        inputs, labels, _ = data
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        true_labels = labels.cpu().data.numpy()
        batch_size = true_labels.shape

        outputs = model(inputs)
        probs = outputs.cpu().data.numpy()

        # get predictions and true values for each item in batch
        for j in range(0, batch_size[0]):
            thisrow = {}
            truerow = {}
            thisrow[FILENAME] = dataset.df.index[BATCH_SIZE * i + j]

            # iterate over each entry in prediction vector; each corresponds to
            # individual label
            thisrow["prob_"+FINDING] = probs[j, 0] #last 0 is multilabel index...here 1
            thisrow[FINDING] = true_labels[j, 0]

            pred_df = pred_df.append(thisrow, ignore_index=True)

        rids = [x[0:x.find("_")] for x in pred_df[FILENAME]] 
        pred_df['rid']=rids
            

    auc_df = pd.DataFrame()

    
    column= str(FINDING)
    actual = pred_df[column]
    pred = pred_df["prob_" + column]
    thisrow = {}
    thisrow['label'] = column
    
    thisrow['auc-image-level'] = sklm.roc_auc_score(actual.to_numpy().astype(int), pred.to_numpy())

    patientlevel=pred_df[['rid',FINDING,'prob_'+FINDING]]
    patientlevel=patientlevel.groupby('rid').max().reset_index()
    
    thisrow['auc-study-max'] = sklm.roc_auc_score(patientlevel[FINDING].to_numpy().astype(int), patientlevel['prob_'+FINDING].to_numpy())

    patientmean=pred_df[['rid','prob_'+FINDING]]
    patientmean=patientmean.groupby('rid').mean().reset_index()
    patientmean=patientmean.merge(patientlevel[['rid',FINDING]],on="rid",how="left")
    thisrow['auc-study-mean'] = sklm.roc_auc_score(patientmean[FINDING].to_numpy().astype(int), patientmean['prob_'+FINDING].to_numpy())
    auc_df = auc_df.append(thisrow, ignore_index=True)

    print("lengths total test and pid  groups "+str(len(pred_df))+" "+str(len(patientlevel))+" "+str(len(patientmean)))

    pred_df.to_csv("results/preds.csv", index=False)
    auc_df.to_csv("results/aucs.csv", index=False)
    print(auc_df)
    return pred_df, auc_df
