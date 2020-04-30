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
from OURS.dataset_parallel import OURDataset

###########################################
# Command line interface
this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
default_out = os.path.join(os.path.dirname(this_dir), "results_ours.csv")
default_ann = "data/human_complete.sga"
default_mod = "models/cnnprom/model.pt"

parser = argparse.ArgumentParser(description=r"This script will test a model's performance with OUR dataset")
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
parser.add_argument('--annotations',
        type = str,
        help = f'Path to desired annotations file. Default: {default_ann}.'
        'The annotations file is an sga obtained from Mass Genome Annotation Data Repository',
        default = default_ann
        )
parser.add_argument('--chromosome', 
        type=str, 
        help='Chromosome name, e.g. 1,2,3..22,X,Y. Only one chromosome can be tested at a time.',
        default="1")
parser.add_argument('--p_us_length', 
        type=int, 
        help='Promoter upstream length for the annotations',
        default=249)
parser.add_argument('--p_ds_length', 
        type=int, 
        help='Promoter downstream length for the annotations',
        default=50)
parser.add_argument('--input_length', 
        type=int, 
        help='Length of the input sequences to be tested. Will be the size of the window',
        default=300)
parser.add_argument('--threshold', 
        type=int, 
        help='Threshold for the number of consecutive nucleotides to be matched to a promoter for a sequence to be classified as a promoter.',
        default=250)
parser.add_argument('--stride', 
        type=int, 
        help='Stride of the window for selecting sequences in the chromosome.'
        'As a guideline, this should be the input length - threshold for good coverage',
        default=50)
args = parser.parse_args()
###########################################

model_folder = os.path.dirname(args.model)
cp = Checkpoint(dirname=model_folder, f_params=os.path.basename(args.model))
# Binary(sigmoid): Use NeuralNetBinaryClassifier, num_classes=1, criterion=BCEWithLogitsLoss, binary=True
# Multi(softmax): Use NeuralNetClassifier, num_classes=2, criterion=CrossEntropyLoss, binary=False

ds = OURDataset(annotations_file=args.annotations,
                chromosome_name=args.chromosome,
                promoter_upstream_length=args.p_us_length,
                promoter_downstream_length=args.p_ds_length,
                input_length=args.input_length,
                threshold=args.threshold,
                stride=args.stride,
                binary=args.binary,
                save_df=False)

module_name = model_folder.split('/')[1]
if(module_name == 'dprom'):
    from DPROM.module import DPROMModule
    module = DPROMModule
elif(module_name == 'cnnprom'):
    from CNNPROM.module import CNNPROMModule
    module = CNNPROMModule

def tqdm_iterator(dataset, **kwargs):
    return tqdm(torch.utils.data.DataLoader(dataset, **kwargs))
if(module_name == 'icnn'):
    from ICNN.module import ICNNModule
    net = NeuralNetClassifier(module=ICNNModule,
                              module__num_classes=2,
                              module__elements_length=ds.elements_length,
                              module__non_elements_length=ds.non_elements_length,
                              batch_size=256,
                              device='cuda' if torch.cuda.is_available() else 'cpu',
                              iterator_valid=tqdm_iterator)
else:
    net = NeuralNetClassifier(module=module,
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