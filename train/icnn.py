import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint

sys.path.append(os.path.join(sys.path[0], '..'))
from ICNN.module import ICNNModule
from ICNN.dataset import ICNNDataset

model_folder = "models/icnn/"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier (!IMPORT IT), num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier (!IMPORT IT), num_classes=2, criterion=CrossEntropyLoss, binary=False

ds = ICNNDataset(file="data/human_representative.fa", neg_folder="data/bdgp", num_positives=7156, binary=False, save_df=True)
print("Preprocessing: Preparing for stratified sampling")
y_train = np.array([y for _, y in tqdm(iter(ds))])
print("Preprocessing: Done")
net = NeuralNetClassifier(module=ICNNModule,
                          module__num_classes=2,
                          module__elements_length=ds.elements_length,
                          module__non_elements_length=ds.non_elements_length,
                          criterion=torch.nn.CrossEntropyLoss,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=5),
                                     ProgressBar(),
                                     Checkpoint(dirname=model_folder,
                                                f_params='model.pt')],
                          batch_size=8,
                          optimizer=torch.optim.SGD,
                          optimizer__momentum=0.90,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
print("Training: Started")
net.fit(ds, y_train)
print("Training: Done")