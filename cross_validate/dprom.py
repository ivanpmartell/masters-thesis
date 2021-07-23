import os
import sys
import torch
import numpy as np
import argparse
from numpy import average
from tqdm import tqdm
from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier
from skorch.dataset import CVSplit
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_validate
from multiscorer.multiscorer import MultiScorer

sys.path.append(os.path.join(sys.path[0], '..'))
from DPROM.module import DPROMModule
from DPROM.dataset import DPROMDataset

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
default_out = os.path.join(os.path.dirname(this_dir), "results.csv")
default_input = "data/human_complete.fa"
default_neg = ""
default_mod = "models/dprom/model.pt"

parser = argparse.ArgumentParser(description=r"This script will test a model's performance with DProm dataset")
parser.add_argument('-binary', 
        type=bool, 
        help='For model: a 1 neuron sigmoid output if set, otherwise a 2 neuron softmax output',
        default=False)
parser.add_argument('--output',
        type = str,
        help = f'Path for desired output file. Default: {default_out}. '
        'The output file is a csv with the sequences tested, their true labels, and the predictions by the model',
        default = default_out
        )
parser.add_argument('--input',
        type = str,
        help = f'Path to desired annotations file. Default: {default_input}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_input
        )
parser.add_argument('--neg_file',
        type = str,
        help = f'Path to desired annotations file. Default: Empty String.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_neg
        )
args = parser.parse_args()
###########################################

model_folder = args.output
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier (!IMPORT IT), num_classes=1, binary=True
# Multi(softmax): Use NeuralNetClassifier (!IMPORT IT), num_classes=2, binary=False

neg_f = None
if(args.neg_file != ''):
    neg_f = args.neg_file
if(args.binary):
    nc = 1
    crit = torch.nn.BCEWithLogitsLoss
    cls = NeuralNetBinaryClassifier
else:
    nc = 2
    crit = torch.nn.CrossEntropyLoss
    cls = NeuralNetClassifier
ds = DPROMDataset(file=args.input, neg_file=neg_f, binary=args.binary, save_df=None, drop_dups=False)
print("Preprocessing: Preparing for stratified sampling")
data_list = [(x, y) for x, y in tqdm(iter(ds))]
X = np.array([col[0] for col in data_list], dtype=np.float32)
y = np.array([col[1] for col in data_list], dtype=np.int64)
print("Preprocessing: Done")
net = cls(module=DPROMModule,
                          module__num_classes=nc,
                          module__seqs_length=ds.seqs_length,
                          criterion=crit,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=10),
                                     ProgressBar()],
                          batch_size=32,
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

with open(os.path.join(model_folder, "cv_results.txt"), 'w') as f:
    for metric in results['confusion matrix'][0].keys():
        f.write("%s: %s\n" % (metric, average(results['confusion matrix'][0][metric])))
    f.write("\n\n")
    f.write(str(results))
print("Results written to file")