# PyTorch model and training necessities
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

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
from utils.spatial_transforms import *
from utils.temporal_transforms import *
from utils.target_transforms import ClassLabel, VideoID
from utils.target_transforms import Compose as TargetCompose
from utils.utils import adjust_learning_rate
from utils.train import *
from utils.validate import *

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

# Build Model
# Generate resnext50 model
# only using 300 labels for now
model = resnxt.resnext50(
                num_classes=300,
                shortcut_type="B",
                cardinality=32,
                sample_size=16, # I have no idea how to decide this rn
                sample_duration=16) # Num frames
parameters = model.parameters() # get complete parameters of resNext model
print(model)

# Loss criterion
criterion = nn.CrossEntropyLoss()

# For training
# crop_method = ['random', 'corner', 'center']
crop_method = 'center'
spatial_transform = Compose([ # figure all this, its from the main of the reference
    #RandomHorizontalFlip(),
    #RandomRotate(),
    #RandomResize(),
    crop_method,
    #MultiplyValues(),
    #Dropout(),
    #SaltImage(),
    #Gaussian_blur(),
    #SpatialElasticDisplacement(),
    # ToTensor(opt.norm_value), norm_method # figure this out
])
temporal_transform = TemporalRandomCrop(16, 1)
target_transform = ClassLabel()
#load data set and select training data
train_data = dataset.get_train_set(spatial_transform, temporal_transform, target_transform)
train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=8,
            shuffle=True,
            num_workers=1, # for now since I only have my pc
            pin_memory=True)
train_logger = Logger(
    os.path.join("./logs", '_train.log'),
    ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
train_batch_logger = Logger(
    os.path.join("./logs", 'train_batch.log'),
    ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

optimizer = optim.SGD(
            parameters,
            lr=0.01,
            momentum=0.9,
            dampening=0.9,
            weight_decay=0.001,
            nesterov=True)
scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                'min', 
                patience=1 # for test purposes
            )

val_data = dataset.get_validation_set(spatial_transform, temporal_transform, target_transform)
val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=8,
            shuffle=False,
            num_workers=1, # for now since I only have my pc
            pin_memory=True)

# BEGIN
best_prec1 = 0
print('run')
n_epochs = 5
for i in range(n_epochs):
# for i in range(opt.begin_epoch, opt.begin_epoch + 10):
    #adjust_learning_rate(optimizer, i) # disregarded bc im not training past 30 eps rn
    train_epoch(i, train_loader, model, criterion, optimizer, None,
                train_logger, train_batch_logger)
    state = {
        'epoch': i,
        'arch': 'n/a',
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_prec1': best_prec1
        }
    save_checkpoint(state, False)