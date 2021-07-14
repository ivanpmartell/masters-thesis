import os
import sys
import torch
import numpy as np
from numpy import average
from tqdm import tqdm
from skorch import NeuralNetClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_score
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
data_list = [(x, y) for x, y in tqdm(iter(ds))]
X = np.array([col[0] for col in data_list], dtype=np.float32)
y = np.array([col[1] for col in data_list], dtype=np.int64)
print("Preprocessing: Done")
net = NeuralNetClassifier(module=CNNPROMModule,
                          module__num_classes=2,
                          module__seqs_length=ds.seqs_length,
                          criterion=torch.nn.CrossEntropyLoss,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=10),
                                     ProgressBar()],
                          batch_size=16,
                          optimizer=torch.optim.Adam,
                          train_split=CVSplit(cv=0.1,stratified=True),
                          device='cuda' if torch.cuda.is_available() else 'cpu')

print("Cross Validation: Started")
#scoring metrics can be modified. Predefined metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
def confusion_matrix_scorer(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    tn = cm[0, 0]
    tp = cm[1, 1]
    fn = cm[1, 0]
    fp = cm[0, 1]
    mcc = matthews_corrcoef(y, y_pred)
    return {'tn': tn , 'fp': fp,
            'fn': fn, 'tp': tp,
            'sensitivity': tp/(tp+fn), 'specificity': tn/(tn+fp),
            'precision': tp/(tp+fp), 'mcc': mcc }
scorer = MultiScorer({
  'confusion matrix': (confusion_matrix_scorer, {})
})
cross_validate(net, X, y, scoring=scorer, cv=10, verbose=1)
print("Cross Validation: Done")
results = scorer.get_results()

for metric in results['confusion matrix'].keys():
  print("%s: %s" % (metric, average(results[metric])))
