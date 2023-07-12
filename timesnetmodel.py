import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from embedding import DataEmbedding


class Block():
    def __init__(self):
        pass


class Model(nn.Module):
    def __init__(self, in_size, out_size):
        super(Model, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.l1 = nn.Sequential(nn.Linear(in_size[0] * 64, out_size[0] * out_size[1]))
        self.embedding = DataEmbedding(c_in=in_size[-1])
    
    def forward(self, x, x_mark):
        x = self.embedding(x, x_mark)
        print(x.dtype)
        y = self.l1(x.reshape(-1, self.in_size[0] * 64))
        return y.reshape(-1, self.out_size[0], self.out_size[1])


class Timesnet(pl.LightningModule):
    def __init__(self, in_size, out_size):
        super(Timesnet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.model = Model(in_size=self.in_size, out_size=self.out_size)


    def training_step(self, batch, batch_idx):
        x, x_mask, y, y_mask = batch
        y_hat = self.model(x, x_mask)
        return F.mse_loss(y_hat, y)


    def validation_step(self):
        pass


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


