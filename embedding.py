import math
import torch
from torch import nn

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model=64):
        super(TokenEmbedding, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode="circular", bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)
    

class PositionEmbedding(nn.Module):
    def __init__(self, d_model=64, max_len=5000):
        super(PositionEmbedding, self).__init__()
        
        div = (torch.arange(0, d_model, 2) / d_model * math.log(10000.0)).exp()
        t = torch.arange(0, max_len).unsqueeze(1)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(t / div)
        pe[:, 1::2] = torch.cos(t / div)
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model=64):
        super(FixedEmbedding, self).__init__()
        
        div = (torch.arange(0, d_model, 2) / d_model * math.log(10000.0)).exp()
        t = torch.arange(0, c_in).unsqueeze(1)
        w = torch.zeros(c_in, d_model)
        w.requires_grad = False
        w[:, 0::2] = torch.sin(t / div)
        w[:, 1::2] = torch.cos(t / div)

        self.emb = nn.Embedding(num_embeddings=c_in, embedding_dim=d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)


    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model=64, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        
        if embed_type == "fixed":
            Emb = FixedEmbedding
        else:
            Emb = nn.Embedding
        if freq == 't':
            self.minute_emb = Emb(c_in=4, d_model=d_model)
        self.hour_emb = Emb(c_in=24, d_model=d_model)
        self.day_emb = Emb(c_in=32, d_model=d_model)
        self.week_emb = Emb(c_in=7, d_model=d_model)
        self.month_emb = Emb(c_in=13, d_model=d_model)

    def forward(self, x):
        x = x.long()
        if hasattr(self, 'minute_emb'):
            minute_x = self.minute_emb(x[:, :, 4])
        else:
            minute_x = 0
        hour_x = self.hour_emb(x[:, :, 3])
        day_x = self.day_emb(x[:, :, 2])
        week_x = self.week_emb(x[:, :, 1])
        month_x = self.month_emb(x[:, :, 0])
        print(minute_x)
        return minute_x + hour_x + day_x + week_x + month_x
    

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model=64, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        
        self.value_emb = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_emb = PositionEmbedding(d_model=d_model)
        self.temporal_emb = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        y = self.value_emb(x) + self.position_emb(x)
        if x_mark != None:
            y += self.temporal_emb(x_mark)
        return self.dropout(y)
