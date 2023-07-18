from typing import Any
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from embedding import DataEmbedding


def FFT(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class Block(nn.Module):
    def __init__(self, k, d_model, d_ff, num_kernels):
        super(Block, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=num_kernels, padding=(num_kernels - 1) // 2), 
            nn.GELU(),
            nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=num_kernels, padding=(num_kernels - 1) // 2)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            if T % period != 0:
                l = ((T // period) + 1) * period
                padding = torch.zeros((B, l - T, N))
                out = torch.cat([x, padding], dim=1)
            else:
                out = x

            out = out.reshape(B, -1, period, N).permute(0, 3, 1, 2).contiguous()
            
            out = self.conv(out)

            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])

        res = torch.stack(res, dim=-1)
        period_weight = torch.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1)
        res = torch.sum(res * period, dim=-1)

        return res + x


class Model(nn.Module):
    def __init__(self, seq, pred, fin, fout, d_model=64, d_ff=64, layers=5, k=5, kernel_size=3):
        super(Model, self).__init__()
        self.embedding = DataEmbedding(c_in=fin)
        self.predict_linear = nn.Linear(seq, pred)    

        self.layers = layers
        self.blocks = []
        for i in range(layers):
            self.blocks.append(Block(k=k, d_model=d_model, d_ff=d_ff, num_kernels=kernel_size))
        
        self.projection = nn.Linear(d_model, fout)

    def forward(self, x, x_mark):
        x = self.embedding(x, x_mark)
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layers):
            x = self.blocks[i](x)    

        y = self.projection(x)
        return y


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
