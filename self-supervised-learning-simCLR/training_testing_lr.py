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

from simclr_model import SimCLR
from logistic_regression import LogisticRegression

DEVICE = torch.device('mps')
DATASET_PATH = '/Users/hoanglinh96nl'
CHECKPOINT_PATH = 'saved_model'

@torch.no_grad()
def prepare_data_features(model: SimCLR, dataset):
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # used to remove projection head
    network.eval()
    network.to(DEVICE)
    
    # encode images
    dataloader = DataLoader(dataset, batch_size=64, num_workers=9, persistent_workers=True, 
                            shuffle=False, drop_last=False)
    features, labels = [], []
    for batch_imgs, batch_labels in tqdm(dataloader):
        batch_imgs = batch_imgs.to(DEVICE)
        batch_features = network(batch_imgs)
        features.append(batch_features.detach().cpu())
        labels.append(batch_labels)
        
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    labels, idxs = labels.sort()
    features = features[idxs]
    return TensorDataset(features, labels)


if __name__ == '__main__':
    # %% Testing data preparation
    test_img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5,))])
    train_img_data = CIFAR100(root=DATASET_PATH, train=True, download=True, transform=test_img_transforms)
    test_img_data = CIFAR100(root=DATASET_PATH, train=False, download=True, transform=test_img_transforms)
    
    # %% Prepare and load model from checkpoint
    pretrained_model_path = 'saved_model/SimCLR/lightning_logs/version_2/checkpoints/epoch=1-step=390.ckpt'
    simclr_model = SimCLR.load_from_checkpoint(pretrained_model_path)
    
    # %% Use SimCLR model as a feature extractor
    train_features_simclr = prepare_data_features(model=simclr_model, dataset=train_img_data)
    test_features_simclr = prepare_data_features(model=simclr_model, dataset=test_img_data)

    # %% Training Logistic Regression model by using extracted features
    trainer = pl.Trainer(default_root_dir=Path(CHECKPOINT_PATH, 'LogisticRegression'), 
                         accelerator='gpu', devices=1, max_epochs=10,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor('epoch')])
    lr_model = LogisticRegression(feature_dim=train_features_simclr.tensors[0].shape[1],
                                  num_classes=10, learning_rate=1e-3, weight_decay=1e-3, max_epochs=10)
    
    train_dataloader = DataLoader(train_features_simclr, batch_size=128, shuffle=True,
                                  drop_last=False, pin_memory=True, num_workers=9, persistent_workers=True)
    test_dataloader = DataLoader(test_features_simclr, batch_size=128, shuffle=False,
                                 drop_last=False, pin_memory=True, num_workers=9, persistent_workers=True)
    trainer.fit(lr_model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
    
    # %% Evaluate LR model
    train_results = trainer.test(lr_model, dataloaders=train_dataloader)
    test_results = trainer.test(lr_model, dataloaders=test_dataloader)
    result = {'train': train_results[0]['test_acc'], 'test': test_results[0]['test_acc']}
    
    