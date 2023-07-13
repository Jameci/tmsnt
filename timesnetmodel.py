from typing import Any
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
        self.l1 = nn.Sequential(nn.Linear(out_size[0] * 64, out_size[0] * out_size[1]))
        self.embedding = DataEmbedding(c_in=in_size[-1])
        self.predict_linear = nn.Linear(in_size[0], out_size[0])

    def forward(self, x, x_mark):
        x = self.embedding(x, x_mark)
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.l1(torch.flatten(x, start_dim=1))
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
        loss = F.mse_loss(y_hat, y)
        self.log("loss", loss)
        result = {
            'loss' : loss
        }
        return result

    def validation_step(self, batch, batch_idx):
        x, x_mask, y, y_mask = batch
        y_hat = self.model(x, x_mask)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        result = {
            'val_loss' : loss
        }
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x, x_mark):
        return self.model(x, x_mark)
