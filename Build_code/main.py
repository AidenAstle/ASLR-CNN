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

model_selection = Model_Select()

