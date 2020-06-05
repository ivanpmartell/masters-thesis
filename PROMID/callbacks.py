from skorch.callbacks import Callback
import torch

class AppendToDataset(Callback):

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None):
        net.module_.eval()
        with torch.no_grad():
            dataset_train.append_false_positive_seqs(net)