import numpy as np
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping
from OURS.dataset import OURDataset
from DPROM.module import DPROMModule
from sklearn.model_selection import cross_validate
from skorch.callbacks import EarlyStopping, ProgressBar
import torch
from tqdm import tqdm

#Copy and paste any dataset (ds) and network (net) from a training file
#Be sure to remove train_split from the net parameters as shown below
ds = OURDataset(annotations_file="data/human_complete.sga",
                chromosome_name="1",
                promoter_upstream_length=249,
                promoter_downstream_length=50,
                input_length=300,
                threshold=250,
                stride=50,
                binary=False)
net = NeuralNetClassifier(module=DPROMModule,
                          module__num_classes=2,
                          module__seqs_length=ds.seqs_length,
                          criterion=torch.nn.CrossEntropyLoss,
                          max_epochs=50,
                          lr=0.001,
                          callbacks=[EarlyStopping(patience=5),
                                     ProgressBar()],
                          batch_size=32,
                          optimizer=torch.optim.Adam,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
print("Cross Validation: Started")
#scoring metrics can be modified. Predefined metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
scores = cross_validate(net, ds.dataframe, ds.dataframe_labels, scoring='accuracy', cv=10, verbose=1)
print("Cross Validation: Done")
print(scores)