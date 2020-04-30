import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint

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

model_folder = os.path.dirname(args.model)
cp = Checkpoint(dirname=model_folder, f_params=os.path.basename(args.model))
# Binary(sigmoid): Use NeuralNetBinaryClassifier, num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier, num_classes=2, criterion=CrossEntropyLoss, binary=False

neg_f = None
if(args.neg_file != ''):
    neg_f = args.neg_file
ds = DPROMDataset(file=args.input, neg_file=neg_f, binary=args.binary, save_df=False)

def tqdm_iterator(dataset, **kwargs):
    return tqdm(torch.utils.data.DataLoader(dataset, **kwargs))

net = NeuralNetClassifier(module=DPROMModule,
                          module__num_classes=2,
                          module__seqs_length=ds.seqs_length,
                          batch_size=256,
                          device='cuda' if torch.cuda.is_available() else 'cpu',
                          iterator_valid=tqdm_iterator)
net.initialize()
print("Testing: Initialized")
net.load_params(checkpoint=cp)
print("Testing: Model Loaded")
print("Testing: Predicting")
y_score = net.predict_proba(ds)
print("Testing: Predicting Done")
if(args.binary):
    df = pd.DataFrame(list(zip(ds.dataframe.sequence, ds.dataframe.label, y_score)), columns=['sequence', 'label', 'prediction'])
else:
    logits = list(zip(*y_score))
    df = pd.DataFrame(list(zip(ds.dataframe.sequence, ds.dataframe.label, logits[0], logits[1])), columns=['sequence', 'label', 'prediction_0', 'prediction_1'])
print("Testing: Saving Results")
df.to_csv(args.output)