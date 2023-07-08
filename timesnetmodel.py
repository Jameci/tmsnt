import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F


class Block():
    def __init__(self):
        pass


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    
    def forward(self, x, x_mark):
        pass


class Timesnet(pl.LightningModule):
    def __init__(self):
        super(Timesnet, self).__init__()
        self.model = Model()


    def training_step(self, batch, batch_idx):
        x, x_mask, y, y_mask = batch
        y_hat = self.model(x, x_mask)
        return F.mse_loss(y_hat, y)


    def validation_step(self):
        pass


    def configure_optimizers(self):
        pass




