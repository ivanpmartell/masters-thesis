from skorch.callbacks import Callback
import torch

class AppendToDataset(Callback):
    counter: int

    def __init__(self):
        self.counter = 0

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None):
        self.counter += 1
        print("Learning rate: %f" % net.get_params()['lr'])
        if net.history[-1]['valid_loss'] > 100:
            net.set_params(lr=0.00001)
        elif net.history[-1]['valid_loss'] > 1:
            net.set_params(lr=0.0005)
        if net.history[-1]['valid_loss'] < 1 or self.counter > 100:
            self.counter = 0
            net.set_params(lr=0.0001)
            net.module_.eval()
            with torch.no_grad():
                dataset_train.append_false_positive_seqs(net)