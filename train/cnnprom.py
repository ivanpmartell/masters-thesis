import os
import sys
import torch
import argparse
import numpy as np
from tqdm import tqdm
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier
from skorch.callbacks import EarlyStopping, ProgressBar, Checkpoint

sys.path.append(os.path.join(sys.path[0], '..'))
from CNNPROM.module import CNNPROMModule
from CNNPROM.dataset import CNNPROMDataset

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
default_out = this_dir
default_input = "data/human_complete.fa"
default_neg = ""
default_neg_size = 27731 + 8256

parser = argparse.ArgumentParser(description=r"This script will test a model's performance with CNNProm dataset")
parser.add_argument('-binary', 
        type=bool, 
        help='For model: a 1 neuron sigmoid output if set, otherwise a 2 neuron softmax output',
        default=False)
parser.add_argument('--output',
        type = str,
        help = f'Path for desired output folder. Default: {default_out}. '
        'The output file is a csv with the sequences tested, their true labels, and the predictions by the model',
        default = default_out
        )
parser.add_argument('--input',
        type = str,
        help = f'Path to desired input file. Default: {default_input}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_input
        )
parser.add_argument('--neg_file',
        type = str,
        help = f'Path to desired annotations file. Default: {default_neg}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_neg
        )
parser.add_argument('--neg_sample',
        type = int,
        help = f'Path to desired annotations file. Default: {default_neg_size}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_neg_size
        )
args = parser.parse_args()
###########################################

model_folder = args.output
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
# Binary(sigmoid): Use NeuralNetBinaryClassifier (!IMPORT IT), num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier (!IMPORT IT), num_classes=2, criterion=CrossEntropyLoss, binary=False

neg_f = None
if(args.neg_file != ''):
    neg_f = args.neg_file
if(args.binary):
    nc = 1
    crit = torch.nn.BCEWithLogitsLoss
else:
    nc = 2
    crit = torch.nn.CrossEntropyLoss
ds = CNNPROMDataset(file=args.input, neg_file=neg_f, num_negatives=args.neg_sample, binary=args.binary, save_df=False)
print("Preprocessing: Preparing for stratified sampling")
y_train = np.array([y for _, y in tqdm(iter(ds))])
print("Preprocessing: Done")
net = NeuralNetBinaryClassifier(module=CNNPROMModule,
                          module__num_classes=nc,
                          module__seqs_length=ds.seqs_length,
                          criterion=crit,
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