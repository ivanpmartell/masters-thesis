import os
import sys
from skorch.classifier import NeuralNetBinaryClassifier
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint

sys.path.append(os.path.join(sys.path[0], '..'))
from CNNPROM.module import CNNPROMModule
from CNNPROM.dataset import CNNPROMDataset

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
default_out = os.path.join(os.path.dirname(this_dir), "results.csv")
default_input = "data/human_complete.fa"
default_neg = ""
default_neg_size = 27731 + 8256
default_mod = "models/cnnprom/model.pt"
default_num_channels = 300
default_pool_size = 231

parser = argparse.ArgumentParser(description=r"This script will test a model's performance with CNNProm dataset")
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
parser.add_argument('--num_channels',
        type = int,
        help = f'Path to desired annotations file. Default: {default_num_channels}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_num_channels
        )
parser.add_argument('--pool_size',
        type = int,
        help = f'Path to desired annotations file. Default: {default_pool_size}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_pool_size
        )
args = parser.parse_args()
###########################################

model_folder = os.path.dirname(args.model)
cp = Checkpoint(dirname=model_folder, f_params=os.path.basename(args.model))
# Binary(sigmoid): Use NeuralNetBinaryClassifier (!IMPORT IT), num_classes=1, binary=True
# Multi(softmax): Use NeuralNetClassifier (!IMPORT IT), num_classes=2, binary=False

neg_f = None
if(args.neg_file != ''):
    neg_f = args.neg_file
if(args.binary):
    nc = 1
    cls = NeuralNetBinaryClassifier
else:
    nc = 2
    cls = NeuralNetClassifier
ds = CNNPROMDataset(file=args.input, neg_file=neg_f, num_negatives=args.neg_sample, binary=args.binary, save_df=None, drop_dups=False)

def tqdm_iterator(dataset, **kwargs):
    return tqdm(torch.utils.data.DataLoader(dataset, **kwargs))

net = cls(module=CNNPROMModule,
                          module__num_classes=nc,
                          module__seqs_length=ds.seqs_length,
                          module__num_channels=args.num_channels,
                          module__pool_kernel=args.pool_size,
                          batch_size=256,
                          device='cuda' if torch.cuda.is_available() else 'cpu',
                          iterator_valid=tqdm_iterator)
net.initialize()
print("Testing: Initialized")
net.load_params(checkpoint=cp)
print("Testing: Model Loaded")
print("Testing: Predicting")
y_score = net.forward(ds)
print("Testing: Predicting Done")
if(args.binary):
    df = pd.DataFrame(list(zip(ds.dataframe.sequence, ds.dataframe.label, y_score.tolist())), columns=['sequence', 'label', 'prediction'])
else:
    logits = list(zip(*y_score.tolist()))
    df = pd.DataFrame(list(zip(ds.dataframe.sequence, ds.dataframe.label, logits[0], logits[1])), columns=['sequence', 'label', 'prediction_0', 'prediction_1'])
print("Testing: Saving Results")
df.to_csv(args.output)