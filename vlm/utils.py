import os
import time
import numpy as np
import matplotlib.pyplot as plt

import medmnist
from medmnist import INFO

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_cifar():
    train = datasets.CIFAR10(root="data", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    val = datasets.CIFAR10(root="data", train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
    return train, val


def load_block(data_flag: str, download: bool=True, clip_bool: bool=False):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    data_file_path = '/data/.medmnist'
    os.makedirs(data_file_path, exist_ok=True)

    data_transform = transforms.Compose([transforms.ToTensor()])

    train = DataClass(root=data_file_path, 
                      split='train',
                      download=download,
                      transform=data_transform,
                      as_rgb=True if info["n_channels"] == 1 else False,
                      size=224)
   
    val = DataClass(root=data_file_path,
                    split='val',
                    download=download,
                    transform=data_transform,
                    as_rgb=True if info["n_channels"] == 1 else False,
                    size=224)
    
    test = DataClass(root=data_file_path,
                    split='test',
                    download=download,
                    transform=data_transform,
                    as_rgb=True if info["n_channels"] == 1 else False,
                    size=224)
    
    return train, val, test

def data_loaders(train_data, val_data, test_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)

    test_loader = DataLoader(test_data,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True)
    
    return train_loader, val_loader, test_loader


def load_data_and_data_loaders(dataset, data_flag, batch_size):
    if dataset == 'CIFAR10':
        training_data, validation_data = load_cifar()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size)
        x_train_var = np.var(training_data.train_data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data, test_data = load_block(data_flag)
        training_loader, validation_loader, test_loader = data_loaders(
            training_data, validation_data, test_data, batch_size)

        x_train_var = np.var(training_loader.dataset.imgs)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, test_data, training_loader, validation_loader, test_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()

def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    os.makedirs(SAVE_MODEL_PATH + "/vlm_data", exist_ok=True)
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vlm_data_' + timestamp + '.pth')
    
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
