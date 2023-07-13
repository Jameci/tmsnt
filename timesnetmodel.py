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
    def __init__(self, seq, pred, fin, fout, d_model=64):
        super(Model, self).__init__()
        self.seq = seq
        self.pred = pred
        self.fin = fin
        self.fout = fout
        self.l1 = nn.Sequential(nn.Linear(pred * d_model, pred * fout))
        self.embedding = DataEmbedding(c_in=fin)
        self.predict_linear = nn.Linear(seq, pred)

    def forward(self, x, x_mark):
        x = self.embedding(x, x_mark)
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.l1(torch.flatten(x, start_dim=1))
        return y.reshape(-1, self.pred, self.fout)


class Timesnet(pl.LightningModule):
    def __init__(self, seq, pred, fin, fout, d_model=64):
        super(Timesnet, self).__init__()
        self.model = Model(seq=seq, pred=pred, fin=fin, fout=fout, d_model=d_model)

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
