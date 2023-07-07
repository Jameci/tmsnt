from dataprovider import Dataset_My
from timesnetmodel import Timesnet
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer

train_dataset = Dataset_My(flag="train")
val_dataset = Dataset_My(flag="val")
train_dataloader = DataLoader(batch_size=128, shuffle=False, dataset=train_dataset)
val_dataloader = DataLoader(batch_size=128, shuffle=False, dataset=val_dataset)

model = Timesnet()

trainer = Trainer()

for x, y, z, w in train_dataloader:
    print(x)
    print(y)
    print(z)
    print(w)
    break

# trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
