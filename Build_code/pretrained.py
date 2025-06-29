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
from utils.tensor_utils import *
from utils.load_utils import *
from utils.utils import *

## Model Selection
def Model_Select():
    usr_selection = int(input("Select a mode:\n\t1. Test (50 gestures) (Enter 1)\n\t2. Using Pretrained Model (Enter 2)\n\t3. Train Model (Enter 3)\n"))
    assert usr_selection in {1, 2, 3}, "[ERROR:] Invalid Selection"
    match usr_selection:
        case 1:
            mode_selected = "tst"
        case 2:
            mod_selected = "pretrain"
        case 3:
            mod_selected = "build"
    print(f"You selected : {mode_selected}")
    return mode_selected

# model_selection = Model_Select()

if __name__ == "__main__":
    
    model_path = "./Build_code/Models/jester_resnext_101_RGB_32.pth"
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        
    #tensorboard writer
    writer = SummaryWriter('./Build_code/logs/tensorBoard')
    
    # Build Model
    # Generate resnext50 model
    # only using 10 labels for now
    model = resnxt.resnext101(
                    num_classes=10,
                    shortcut_type="B",
                    cardinality=32,
                    sample_size=224, # Its the crop essentially
                    sample_duration=16) # Num frames
    # Handles loading pretrained model data
    pretrain = torch.load(model_path, map_location=torch.device(device))
    pretrain["state_dict"].pop('module.fc.weight') # remove final fc layer from state dict
    pretrain["state_dict"].pop('module.fc.bias') # remove final fc layer from state dict
    #model = modify_kernels(model)
    if device == 'cpu': #wrap with parallel real quick to access module data
        model = nn.DataParallel(model, device_ids=None)
        # pretrain.module.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        # pretrain["state_dict"].module.fc = nn.Linear(pretrain["state_dict"].module.fc.in_features, 10) # alter final and first layer to manage size mismatches
        model.load_state_dict(pretrain["state_dict"], strict=False) # loads pretrained model weights
        # model.module.fc = nn.Linear(model.module.fc.in_features, 10) # should be unnecesary
        model = model.module # unwrap from parallel if not using cuda
    else:
        model.module.fc = model.module.fc.cuda()
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        model.load_state_dict(pretrain["state_dict"])
        # model.module.fc = nn.Linear(model.module.fc.in_features, 10)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if
                               p.requires_grad)
        print("Total number of trainable parameters: ", pytorch_total_params)
        
    #debugging
    # print(pretrain["state_dict"])
    # print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    # print(model.state_dict())
    model = modify_kernels(model)
    
    parameters = model.parameters() # get complete parameters of resNext model
    
    print("------------------------------------------------------------- MODEL -------------------------------------------------------------\n", model)
    print("------------------------------------------------------------- PARAMETERS -------------------------------------------------------------\n", parameters)

    # Loss criterion
    criterion = nn.CrossEntropyLoss()
    if device == 'cuda':
        criterion.cuda()
    print(f"------------------------------------------------------------\n :: Running with {device} :: \n------------------------------------------------------------")
    
    
    
    # For training
    # crop_method = ['random', 'corner', 'center']
    # crop_method = CenterCrop(100)
    crop_method = MultiScaleRandomCrop([1.0, 1.0, 1.0, 1.0, 1.0], 224) # using default scales=1.0, n_sclaes=5, and size=112 for simplicity
    spatial_transform = Compose([ # just using simple random cropping for now
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
    # temporal_transform = TemporalRandomCrop(16, 1)
    target_transform = ClassLabel()
    #load data set and select training data
    train_data = dataset.get_train_set(spatial_transform=spatial_transform, temporal_transform=None, target_transform=target_transform, partial_set=True)
    train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=8,
                shuffle=True,
                num_workers=1, # for now since I only have my pc
                collate_fn=collate_function,
                pin_memory=True)
    train_logger = Logger(
        os.path.join("./Build_code/logs", '_train.log'),
        ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
    train_batch_logger = Logger(
        os.path.join("./Build_code/logs", 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])

    optimizer = optim.SGD(
                parameters,
                lr=0.004,
                momentum=0.9,
                dampening=0.9,
                weight_decay=0.001,
                nesterov=False)
    scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    'min', 
                    patience=3 # for test purposes
                )

    val_data = dataset.get_validation_set(spatial_transform=spatial_transform, temporal_transform=None, target_transform=target_transform, partial_set=True)
    val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=8,
                shuffle=False,
                num_workers=1, # for now since I only have my pc
                collate_fn=collate_function,
                pin_memory=True)
    val_logger = Logger(os.path.join('./Build_code/logs', '_val.log'), ['epoch', 'loss', 'prec1', 'prec5'])

    # BEGIN
    best_prec1 = 0
    print('run')
    n_epochs = 10
    for i in range(n_epochs):
    # for i in range(opt.begin_epoch, opt.begin_epoch + 10):
        #adjust_learning_rate(optimizer, i) # disregarded bc im not training past 30 eps rn
        if device == 'cuda':
            cuda_opt = True
        else:
            cuda_opt = False
        train_loss = train_epoch(i, train_loader, model, criterion, optimizer, cuda_opt,
                    train_logger, train_batch_logger)
        state = {
            'epoch': i,
            'arch': 'n/a',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1
        }
        #save_checkpoint(state, False)
        
        # validation
        validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, cuda_opt, val_logger)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        state = {
            'epoch': i,
            'arch': 'n/a',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1
        }
        if is_best:
            save_dict = model.state_dict()
        loss_graph(writer, train_loss, validation_loss, i)

    print('------------------------------------------------------------ :: Shut Down :: ------------------------------------------------------------')
    # torch.save(save_dict, f"./Pretrained/{n_epochs}_trained.pth")
    writer.close()