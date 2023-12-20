import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import STL10, CIFAR100

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tqdm import tqdm

from data_prepration import ContrastiveTransformations, LoadUnlabelData
from simclr_model import SimCLR


DEVICE = torch.device('mps')
DATASET_PATH = '/Users/hoanglinh96nl'
CHECKPOINT_PATH = 'saved_model'


if __name__ == '__main__':
    # %% Data preparation for self-supvervised learning
    contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(size=96),
                                            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                                                            saturation=0.5, hue=0.1)], p=0.8),
                                            transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=9),
                                            transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

    unlabeled_data = CIFAR100(root=DATASET_PATH, download=True, train=True,
                        transform=ContrastiveTransformations(base_transforms=contrast_transforms, n_views=2))
    unlabeled_data = LoadUnlabelData(unlabeled_data)
    train_data_contrast = CIFAR100(root=DATASET_PATH, download=True, train=True,
                                transform=ContrastiveTransformations(base_transforms=contrast_transforms, n_views=2))

    # %% Training loop
    batch_size, max_epochs = 256, 10
    hidden_dim, learning_rate, temperature, weight_decay = 128, 5e-4, 0.07, 1e-4

    trainer = pl.Trainer(default_root_dir=Path(CHECKPOINT_PATH, 'SimCLR'),
                        accelerator='gpu', devices=1, max_epochs=10,
                        callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = None

    train_dataloader = DataLoader(dataset=unlabeled_data, batch_size=batch_size, shuffle=True,
                                    drop_last=True, pin_memory=True, num_workers=9, persistent_workers=True)
    val_dataloader = DataLoader(train_data_contrast, batch_size=batch_size, shuffle=False,
                            drop_last=False, pin_memory=True, num_workers=9, persistent_workers=True)
    simclr_model = SimCLR(hidden_dim=hidden_dim, learning_rate=learning_rate, temperature=temperature,
                            weight_decay=weight_decay, max_epochs=max_epochs)
    trainer.fit(model=simclr_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    
