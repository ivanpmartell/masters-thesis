import os
import sys
import torch
import numpy as np
from numpy import average
from tqdm import tqdm
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from multiscorer.multiscorer import MultiScorer

sys.path.append(os.path.join(sys.path[0], '..'))
from CNNPROM.module import CNNPROMModule
from CNNPROM.dataset import CNNPROMDataset

model_folder = "models/cnnprom/"
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier, num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier, num_classes=2, criterion=CrossEntropyLoss, binary=False

ds = CNNPROMDataset(file="data/human_TATA.fa", neg_file=None , num_negatives=8256, binary=False, save_df=True)
print("Preprocessing: Preparing for stratified sampling")
y_train = np.array([y for _, y in tqdm(iter(ds))])
print("Preprocessing: Done")
net = NeuralNetClassifier(module=CNNPROMModule,
                          module__num_classes=2,
                          module__seqs_length=ds.seqs_length,
                          criterion=torch.nn.CrossEntropyLoss,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=2),
                                     ProgressBar(),
                                     Checkpoint(dirname=model_folder,
                                                f_params='model.pt')],
                          batch_size=16,
                          optimizer=torch.optim.Adam,
                          train_split=CVSplit(cv=0.2,stratified=True))

print("Cross Validation: Started")
#scoring metrics can be modified. Predefined metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
scorer = MultiScorer({
  'accuracy': (accuracy_score, {}),
  'precision': (precision_score, {}),
  'recall': (recall_score, {}),
  'mcc': (matthews_corrcoef, {})
})
cross_validate(net, ds, y_train, scoring=scorer, cv=2, verbose=1)
print("Cross Validation: Done")
results = scorer.get_results()

for metric in results.keys():
  print("%s: %.3f" % (metric, average(results[metric])))