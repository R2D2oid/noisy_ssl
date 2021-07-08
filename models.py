'''
lightly MoCo model
source: https://docs.lightly.ai/tutorials/package/tutorial_moco_memory_bank.html
'''

import pytorch_lightning as pl
import lightly
import torch
from torch import nn

class MocoModel(pl.LightningModule):
    def __init__(self, 
                 backbone_type='resnet-18', 
                 max_epochs=100, 
                 memory_bank_size=4096, 
                 num_ftrs=512, 
                 batch_shuffle=True, 
                 lr=6e-2,
                 momentum=0.9,
                 weight_decay=5e-4,
                 loss_temperature=0.1
                ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_epochs = max_epochs
        
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator(backbone_type, 1, num_splits=8)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        self.resnet_moco = lightly.models.MoCo(backbone, num_ftrs=num_ftrs, m=0.99, batch_shuffle=batch_shuffle)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=loss_temperature,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_moco(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        self.log('train_moco_lr',self.scheduler.optimizer.param_groups[0]['lr'])
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=self.lr,
                                momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [self.scheduler]
    
    
class Classifier(pl.LightningModule):
    def __init__(self, model, lr=30., max_epochs=100):
        super().__init__()
        
        self.lr = lr
        self.max_epochs = max_epochs
        
        # create a moco based on ResNet
        self.resnet_moco = model

        # freeze the layers of moco
        for p in self.resnet_moco.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(512, 10)

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.resnet_moco.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        self.log('train_fc_lr',self.scheduler.optimizer.param_groups[0]['lr'])
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [self.scheduler]