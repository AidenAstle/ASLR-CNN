# PyTorch model and training necessities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Image datasets and image manipulation
import torchvision
import torchvision.transforms as transforms

# Image display
import matplotlib.pyplot as plt
import numpy as np

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

# Local Imports
from Models import resnxt
import dataset

## Model Selection
def Model_Select():
    usr_selection = int(input("Select a model:\n\t1. jester_resnext_101_RGB_32 (Enter 1)\n\t2. (Enter 2)\n\t3. (Enter 3)\n"))
    assert usr_selection in {1, 2, 3}, "[ERROR:] Invalid Selection"
    match usr_selection:
        case 1:
            mod_selected = "resnxt"
        case 2:
            mod_selected = "not_imp"
        case 3:
            mod_selected = "not_imp"
    print(f"You selected : {mod_selected}")
    return mod_selected

# model_selection = Model_Select()

#load data set and select training data
train_data = dataset.get_test_set()
val_data = dataset.get_validation_set()

# Generate resnext50 model
# only using 300 labels for now
model = resnxt.resnext50(
                num_classes=300,
                shortcut_type="B",
                cardinality=32,
                sample_size=16, # I have no idea how to decide this rn
                sample_duration=16)