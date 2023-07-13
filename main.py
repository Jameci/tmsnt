import torch
from dataprovider import Dataset_My
from timesnetmodel import Timesnet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir=r"./log")
writer = SummaryWriter(log_dir=logger.log_dir)

train_dataset = Dataset_My(flag="train")
val_dataset = Dataset_My(flag="val")
train_dataloader = DataLoader(batch_size=128, shuffle=False, dataset=train_dataset)
val_dataloader = DataLoader(batch_size=128, shuffle=False, dataset=val_dataset)

in_size, in_mark, out_size, out_mark = train_dataset.get_size()

model = Timesnet(in_size=in_size, out_size=out_size)

trainer = Trainer(logger=logger, max_epochs=15)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

sameple_in = torch.zeros(1, in_size[0], in_size[1])
sameple_in_mark = torch.zeros(1, in_mark[0], in_mark[1])
writer.add_graph(model, (sameple_in, sameple_in_mark))