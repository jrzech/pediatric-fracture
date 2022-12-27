import rad_dataset as D
import eval_model as E
import model as M
import os

SAMPLE=0
EPOCHS=100
#allows one to specify arguments
print("/n/n/n/n/n/n/n")


IMAGE_SIZE=512
PATH_TO_IMAGES = "/home/jrzech/research/DOCKERSHARED/image/pedue-cropnorm-8bit/"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
AUGMENT=True
CLAHE=False
BATCH_SIZE=8
USE_METADATA=False
BODYPARTS="any"
OPTIMIZER="SGD" #one of SGD, Adam
TARGET=["manual-image-fx"]
preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY,IMAGE_SIZE,AUGMENT,CLAHE,BATCH_SIZE,USE_METADATA,BODYPARTS,SAMPLE,OPTIMIZER,TARGET,EPOCHS)
os.rename("results","results-512-cropnorm-8bit-finelabel")




