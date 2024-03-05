import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms

import math

def Torch_DataLoader( dataset_root_path, batch_size, number_of_workers, download_data ):

    cifar_transforms = transforms.Compose([
        transforms.RandomCrop( (32, 32), padding = 4 ),
        transforms.RandomHorizontalFlip( 0.5 ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4914, 0.4822, 0.4465), 
            std = (0.2023, 0.1994, 0.2010)
        )
    ])

    training_dataset = torchvision.datasets.CIFAR10(
        root = dataset_root_path,
        train = True,
        download = download_data,
        transform = cifar_transforms
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = number_of_workers
    )

    return train_loader


