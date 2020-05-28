import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from skorch import NeuralNetBinaryClassifier
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint

sys.path.append(os.path.join(sys.path[0], '..'))
from CNNPROM.module import CNNPROMModule
from CNNPROM.dataset import CNNPROMDataset

model_folder = "models/cnnprom/"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier (!IMPORT IT), num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier (!IMPORT IT), num_classes=2, criterion=CrossEntropyLoss, binary=False

ds = CNNPROMDataset(file="data/human_TATA.fa", neg_file=None , num_negatives=8256, binary=True, save_df=True)
print("Preprocessing: Preparing for stratified sampling")
y_train = np.array([y for _, y in tqdm(iter(ds))])
print("Preprocessing: Done")
net = NeuralNetBinaryClassifier(module=CNNPROMModule,
                          module__num_classes=1,
                          module__seqs_length=ds.seqs_length,
                          criterion=torch.nn.BCEWithLogitsLoss,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=5),
                                     ProgressBar(),
                                     Checkpoint(dirname=model_folder,
                                                f_params='model.pt')],
                          batch_size=16,
                          optimizer=torch.optim.Adam,
                          device='cuda' if torch.cuda.is_available() else 'cpu')

print("Training: Started")
net.fit(ds, y_train)
print("Training: Done")