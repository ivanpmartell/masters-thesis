import os
import torch
import numpy as np
from tqdm import tqdm
from DPROM.module import DPROMModule
from DPROM.dataset import DPROMDataset
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint
from focal_loss import FocalLoss

model_folder = "models/dprom/"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier, num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier, num_classes=2, criterion=CrossEntropyLoss, binary=False

ds = DPROMDataset(file="data/human_complete.fa", neg_file=None, binary=False, save_df=True)
print("Preprocessing: Preparing for stratified sampling")
y_train = np.array([y for _, y in tqdm(iter(ds))])
print("Preprocessing: Done")
net = NeuralNetClassifier(module=DPROMModule,
                          module__num_classes=2,
                          module__seqs_length=ds.seqs_length,
                          criterion=torch.nn.CrossEntropyLoss,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=5),
                                     ProgressBar(),
                                     Checkpoint(dirname=model_folder,
                                                f_params='model.pt')],
                          batch_size=32,
                          optimizer=torch.optim.Adam,
                          train_split=CVSplit(cv=0.1,stratified=True),
                          device='cuda' if torch.cuda.is_available() else 'cpu')
print("Training: Started")
net.fit(ds, y_train)
print("Training: Done")