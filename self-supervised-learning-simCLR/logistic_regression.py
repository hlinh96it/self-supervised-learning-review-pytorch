import pytorch_lightning as pl
import torch
import torch.nn as nn


class LogisticRegression(pl.LightningModule):
    def __init__(self, feature_dim, num_classes, learning_rate, weight_decay, max_epochs=100):
        super(LogisticRegression, self).__init__()
        self.save_hyperparameters()
        self.model = nn.Linear(feature_dim, num_classes)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, gamma=0.1,
                                                            milestones=[int(self.hparams.max_epochs*0.6),
                                                                        int(self.hparams.max_epochs*0.8)])
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='val')
    
    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='test')
    
    def _calculate_loss(self, batch, mode):
        features, labels = batch
        preds = self.model(features)
        loss = torch.nn.functional.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log(mode + '_loss', loss)
        self.log(mode + '_acc', acc)
        return loss
        