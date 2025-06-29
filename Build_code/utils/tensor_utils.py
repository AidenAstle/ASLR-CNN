# pyTorch training and model imports 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Img datasets and img manipulations
import torchvision
import torchvision.transforms as transforms

# img display
import matplotlib.pyplot as plt
import numpy as np

# Tensorboard Support from pyTorch
from torch.utils.tensorboard import SummaryWriter

# Creates sclar graph of training vs. validation loss
# averaged per epoch
def loss_graph(writer, train_loss, val_loss, epoch):
    writer.add_scalars('Training vs. Validation Loss', 
        {'Training' : train_loss, 'Validation' : val_loss},
        epoch
    )
    writer.flush()