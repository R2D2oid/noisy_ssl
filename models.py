'''
lightly MoCo model
source: https://docs.lightly.ai/tutorials/package/tutorial_moco_memory_bank.html
'''

import pytorch_lightning as pl
import lightly
import torch
from torch import nn

from utils import knn_predict, BenchmarkModule

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
        optim = torch.optim.SGD(self.resnet_moco.parameters(), 
                                lr=self.lr,
                                momentum=self.momentum, 
                                weight_decay=self.weight_decay
                               )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [self.scheduler]
    

class BartonTwins(BenchmarkModule):
    '''
    adopted from: 
            https://github.com/IgorSusmelj/barlowtwins
    '''
    def __init__(self, dataloader_kNN,
                 gpus, 
                 classes=10, 
                 knn_k=200, 
                 knn_t=0.1, 
                 max_epochs=100,
                 backbone_type='resnet-18',
                 num_ftrs=512,
                 num_mlp_layers=3,
                 momentum=0.9,
                 weight_decay=5e-4,
                 lr=1e-3                
                ):
        super().__init__(dataloader_kNN, gpus, classes, knn_k, knn_t)
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.max_epochs = max_epochs
        
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator(backbone_type)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        # note that bartontwins has the same architecture
        self.resnet_simsiam = lightly.models.SimSiam(self.backbone, 
                                                     num_ftrs=num_ftrs, 
                                                     num_mlp_layers=num_mlp_layers)
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = BarlowTwinsLoss(device=device)
            
    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        # our simsiam model returns both (features + projection head)
        z_a, _ = x0
        z_b, _ = x1
        loss = self.criterion(z_a, z_b)
        self.log('train_loss_ssl', loss)
        return loss

    # learning rate warm-up
    def optimizer_steps(self,
                        epoch=None,
                        batch_idx=None,
                        optimizer=None,
                        optimizer_idx=None,
                        optimizer_closure=None,
                        on_tpu=None,
                        using_native_amp=None,
                        using_lbfgs=None):
        # 120 steps ~ 1 epoch
        if self.trainer.global_step < 1000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), 
                                lr=self.lr,
                                momentum=self.momentum, 
                                weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [scheduler]

    

class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, device, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()

        return loss

    
class SimpleResnet(pl.LightningModule):
    def __init__(self, backbone_type='resnet-18'):
        super().__init__()
        
        # create a ResNet backbone and remove the classification head
        self.resnet = lightly.models.ResNetGenerator(backbone_type, 1, num_splits=8)
        self.backbone = nn.Sequential(
            *list(self.resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        
        
class Classifier(pl.LightningModule):
    def __init__(self, model, lr=30., max_epochs=100, freeze_backbone=True):
        super().__init__()
        
        self.lr = lr
        self.max_epochs = max_epochs
        
        # create a moco based on ResNet
        self.resnet_ssl = model

        if freeze_backbone==True:
            # freeze the layers of moco
            for p in self.resnet_ssl.parameters():  # reset requires_grad
                p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(512, 10)

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.resnet_ssl.backbone(x).squeeze()
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
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        self.log('train_fc_lr',self.scheduler.optimizer.param_groups[0]['lr'])
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [self.scheduler]

class Classifier2(pl.LightningModule):
    def __init__(self, model, lr=0.1, max_epochs=100, weight_decay=0.0001, momentum=0.9, freeze_backbone=True):
        super().__init__()

        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay=weight_decay
        self.momentum=momentum

        self.last_acc = 0.0
        self.model=model
        if freeze_backbone==True:
            for p in model.parameters():
                p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(512, 10)

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.model.backbone(x).squeeze()
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
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        self.log('train_fc_lr',self.scheduler.optimizer.param_groups[0]['lr'])
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        self.accuracy(y_hat, y)
        acc = self.accuracy.compute()
        self.log('val_acc', acc,
                 on_epoch=True, prog_bar=True)
        self.last_acc=acc

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                momentum=self.momentum)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.max_epochs)
        return [optim], [self.scheduler]