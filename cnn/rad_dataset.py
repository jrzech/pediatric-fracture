import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms.functional as F
import glob
import cv2

class RadDataset(Dataset):

    def __init__(
            self,
            path_to_images,
            fold,
            transform,
            sample=0,
            clahe=False,
            usemetadata=False,
            bodyparts="any",target=["manual-image-fx"]):

        self.transform = transform
        self.path_to_images = path_to_images
        print(self.path_to_images)
        self.df = pd.read_csv("../../metadata/pedue-deid.csv")
        self.clahe=clahe
        self.metadata=usemetadata
        self.bodyparts=bodyparts
        self.target=target[0]
        print("note - only learning one target: " +self.target) 
        if not self.bodyparts=="any": self.bodyparts=self.bodyparts.split("&")
        self.sample=sample
        print("initial size")
        print(len(self.df))

        if not self.bodyparts=="any":
            self.df = self.df[self.df['Body Part'].isin(self.bodyparts)]
        print("dataset includes these anatomic regions:")
        print(self.df['Body Part'].unique())
        if fold=="test":
            self.df = self.df[((self.df['fold']=="exttest") | (self.df['fold']=="inttest") | (self.df['fold']=="test"))]
        else:
            self.df = self.df[self.df['fold'] == fold]
       
        print("target is "+self.target)
        if self.target!="manual-image-fx":
            print("avg test target before manual replacement")
            print(self.df[self.df['fold'].str.find("test")>=0][self.target].mean())
            self.df['manual-image-fx']=np.where(self.df['fold'].str.find("test")>=0,self.df['manual-image-fx'],np.nan)
        self.df['abnormal'] = np.where(np.isnan(self.df['manual-image-fx']),self.df[self.target],self.df['manual-image-fx'])

    

        if(sample > 0 and sample < len(self.df)):
            if fold!="train": sample=int(sample/5) #make even smaller for tune
            self.df = self.df.sample(sample)

        self.df = self.df.set_index("filename")
        print("exit size")
        print(len(self.df))

        RESULT_PATH = "results/"
        print("***created "+fold+" dataset with size "+str(len(self.df))+" and pos % "+str(self.df['abnormal'].mean())+"***")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = cv2.imread(
                os.path.join(
                    self.path_to_images,
                    self.df.index[idx]),-1)
                

        if(self.clahe):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint16)
            clahe = cv2.createCLAHE(clipLimit = 4,tileGridSize=(12, 12))
            image = clahe.apply(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= image.max()

        label = np.array([int(self.df['abnormal'].iloc[idx])], dtype=np.int)


        if self.transform:
            image = self.transform(image=image)
            image = image["image"]

        return (image, label,self.df.index[idx])


    
