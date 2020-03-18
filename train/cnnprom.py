import os
import torch
import numpy as np
from tqdm import tqdm
from CNNPROM.module import CNNPROMModule
from CNNPROM.dataset import CNNPROMDataset
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint
from sklearn.model_selection import cross_val_score

model_folder = "models/cnnprom/"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier, num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier, num_classes=2, criterion=CrossEntropyLoss, binary=False

ds = CNNPROMDataset(file="data/human_complete.fa", neg_file=None , num_negatives=27731 + 8256, binary=False, save_df=True)
print("Preprocessing: Preparing for stratified sampling")
y_train = np.array([y for _, y in tqdm(iter(ds))])
print("Preprocessing: Done")
net = NeuralNetClassifier(module=CNNPROMModule,
                          module__num_classes=2,
                          module__seqs_length=ds.seqs_length,
                          criterion=torch.nn.CrossEntropyLoss,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=5),
                                     ProgressBar(),
                                     Checkpoint(dirname=model_folder,
                                                f_params='model.pt')],
                          batch_size=16,
                          optimizer=torch.optim.Adam,
                          train_split=CVSplit(cv=0.2,stratified=True))
print("Training: Started")
net.fit(ds, y_train)
print("Training: Done")