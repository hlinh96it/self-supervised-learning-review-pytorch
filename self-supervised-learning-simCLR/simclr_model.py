import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision


class SimCLR(pl.LightningDataModule):
    def __init__(self, hidden_dim, learning_rate, temperature, weight_decay, max_epochs=100):
        super(SimCLR, self).__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, f'The temperature = {temperature} smaller than 0!' 
        
        self.convnet = torchvision.models.resnet18(num_classes=4*hidden_dim)
        self.convnet.fc = nn.Sequential([
            self.convnet.fc,  # Linear(Resnet output, 4*hidden_dim)
            nn.ReLU(inplace=True), nn.Linear(4*hidden_dim, hidden_dim)
        ])
        
    def configure_optimizers(self):
        optimzier = torch.optim.AdamW(params=self.convnet.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimzier, T_max=self.hparams.max_epochs,
                                                                  eta_min=self.hparams.learning_rate/50)
        return [optimzier], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, model='val')
    
    def info_nce_loss(self, batch, mode='train'):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)
        
        features = self.convnet(imgs)  # extract feature by resnet model
        cos_sim = torch.nn.functional.cosine_similarity(features[:, None, :], features[None, :, :], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, value=-9e15)
        
        # Find positive sample -> batch_size // 2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        
        cos_sim = cos_sim / self.hparams.temperature  # infoNCE loss  
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        
        # get ranking positive example
        comb_sim = torch.cat([cos_sim[pos_mask][: None], cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True,).argmin(dim=-1)
                
        # logging loss
        self.log(mode + '_loss', nll)
        self.log(mode + '_acc_top1', (sim_argsort==0).float().mean())
        self.log(mode + '_acc_top5', (sim_argsort < 5).float().mean())
        self.log(mode + '_acc_mean_pos', 1 + sim_argsort.float().mean())
        return nll