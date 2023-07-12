from dataprovider import Dataset_My
from timesnetmodel import Timesnet
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir=r"./log")

train_dataset = Dataset_My(flag="train")
val_dataset = Dataset_My(flag="val")
train_dataloader = DataLoader(batch_size=128, shuffle=False, dataset=train_dataset)
val_dataloader = DataLoader(batch_size=128, shuffle=False, dataset=val_dataset)

in_size, out_size = train_dataset.get_size()

model = Timesnet(in_size=in_size, out_size=out_size)

trainer = Trainer(logger=logger)

trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
