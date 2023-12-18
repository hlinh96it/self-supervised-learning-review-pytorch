import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import STL10

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data_prepration import ContrastiveTransformations
from simclr_model import SimCLR


DEVICE = torch.device('mps')
DATASET_PATH = '/Users/hoanglinh96nl'
CHECKPOINT_PATH = '/Users/hoanglinh96nl/Library/CloudStorage/GoogleDrive-hoanglinh96nl@gapp.nthu.edu.tw/My Drive/Personal Projects/self-supervised-learning-review-pytorch/self-supervised-learning-simCLR'

# %% Data preparation for self-supvervised learning
contrast_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(size=96),
                                          transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                                                                         saturation=0.5, hue=0.1)], p=0.8),
                                          transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=9),
                                          transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

unlabeled_data = STL10(root=DATASET_PATH, split='unlabeled', download=True,
                       transform=ContrastiveTransformations(base_transforms=contrast_transforms, n_views=2))
train_data_contrast = STL10(root=DATASET_PATH, split='train', download=True,
                            transform=ContrastiveTransformations(base_transforms=contrast_transforms, n_views=2))

# %% Training loop
batch_size, max_epochs = 256, 10
hidden_dim, learning_rate, temperature, weight_decay = 128, 5e-4, 0.07, 1e-4

trainer = pl.Trainer(default_root_dir=Path(CHECKPOINT_PATH, 'SimCLR'),
                     accelerator='gpu', devices=1, max_epochs=10,
                     callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
                                LearningRateMonitor('epoch')])
trainer.logger._default_hp_metric = None
pretrained_filename = input('Enter path of pretrained model (Y/N): ')
if pretrained_filename == 'N':
    train_dataloader = DataLoader(dataset=unlabeled_data, batch_size=batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=9)
    val_loader = DataLoader(train_data_contrast, batch_size=batch_size, shuffle=False,
                            drop_last=False, pin_memory=True, num_workers=9)
    simclr_model = SimCLR(hidden_dim=hidden_dim, learning_rate=learning_rate, temperature=temperature, 
                   weight_decay=weight_decay, max_epochs=max_epochs)
    trainer.fit(model=simclr_model, train_dataloaders=train_dataloader, val_dataloaders=val_loader)
    