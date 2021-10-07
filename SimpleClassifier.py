import pytorch_lightning as pl
import lightly
import torch
from torch import nn

class SimpleResnetClassifier(pl.LightningModule):
    def __init__(self, lr=0.01, weight_decay=0.0001, momentum=0.9, max_epochs=100):
        super().__init__()
        
        self.lr=lr
        self.weight_decay=weight_decay
        self.momentum=momentum
        self.max_epochs=max_epochs
        
        self.resnet = lightly.models.ResNetGenerator('resnet-18', num_classes=10) 
        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        y_hat = self.resnet(x).squeeze()
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        self.log('train_fc_lr',self.scheduler.optimizer.param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        self.accuracy(y_hat, y)
        acc = self.accuracy.compute()
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.last_acc=acc

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet.parameters(), 
                                lr=self.lr,
                                momentum=self.momentum, 
                                weight_decay=self.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [self.scheduler]