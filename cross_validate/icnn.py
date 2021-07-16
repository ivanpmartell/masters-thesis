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
from ICNN.module import ICNNModule
from ICNN.dataset import ICNNDataset

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
default_out = os.path.join(os.path.dirname(this_dir), "results.csv")
default_input = "data/human_representative.fa"
default_pos_size = 7156
default_neg = "data/bdgp"
default_mod = "models/icnn/model.pt"

parser = argparse.ArgumentParser(description=r"This script will test a model's performance with ICNN dataset")
parser.add_argument('-binary', 
        type=bool, 
        help='For model: a 1 neuron sigmoid output if set, otherwise a 2 neuron softmax output',
        default=False)
parser.add_argument('--model',
        type = str,
        help = f'Path for desired model file. Default: {default_mod}. '
        'The model file is a checkpoint created by pytorch with the weights of a model',
        default = default_mod
        )
parser.add_argument('--output',
        type = str,
        help = f'Path for desired output file. Default: {default_out}. '
        'The output file is a csv with the sequences tested, their true labels, and the predictions by the model',
        default = default_out
        )
parser.add_argument('--input',
        type = str,
        help = f'Path to desired input file. Default: {default_input}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_input
        )
parser.add_argument('--neg_folder',
        type = str,
        help = f'Path to desired annotations file. Default: {default_neg}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_neg
        )
parser.add_argument('--pos_sample',
        type = int,
        help = f'Path to desired annotations file. Default: {default_pos_size}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_pos_size
        )
args = parser.parse_args()
###########################################

model_folder = os.path.dirname(args.model)
cp = Checkpoint(dirname=model_folder, f_params=os.path.basename(args.model))
# Binary(sigmoid): Use NeuralNetBinaryClassifier (!IMPORT IT), num_classes=1, binary=True
# Multi(softmax): Use NeuralNetClassifier (!IMPORT IT), num_classes=2, binary=False

if(args.binary):
    nc = 1
    crit = torch.nn.BCEWithLogitsLoss
    cls = NeuralNetBinaryClassifier
else:
    nc = 2
    crit = torch.nn.CrossEntropyLoss
    cls = NeuralNetClassifier
ds = ICNNDataset(file=args.input, neg_folder=args.neg_folder, num_positives=args.pos_sample, binary=args.binary, save_df=None)
print("Preprocessing: Preparing for stratified sampling")
data_list = [(x, y) for x, y in tqdm(iter(ds))]
X = [col[0] for col in data_list]
y = np.array([col[1] for col in data_list], dtype=np.int64)
print("Preprocessing: Done")
net = NeuralNetClassifier(module=ICNNModule,
                          module__num_classes=2,
                          module__elements_length=ds.elements_length,
                          module__non_elements_length=ds.non_elements_length,
                          criterion=torch.nn.CrossEntropyLoss,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=10),
                                     ProgressBar()],
                          batch_size=8,
                          optimizer=torch.optim.SGD,
                          optimizer__momentum=0.90,
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
cross_validate(net, ds, y, scoring=scorer, cv=10, verbose=1)
print("Cross Validation: Done")
results = scorer.get_results()

with open(os.path.join(model_folder, "cv_results.txt"), 'w') as f:
    for metric in results['confusion matrix'][0].keys():
        f.write("%s: %s\n" % (metric, average(results['confusion matrix'][0][metric])))
    f.write("\n\n")
    f.write(str(results))
print("Results written to file")