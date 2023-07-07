from dataprovider import Dataset_My
from timesnetmodel import Timesnet
from torch.utils.data import DataLoader


dataset = Dataset_My()
dataloader = DataLoader(batch_size=128, shuffle=False, dataset=dataset)

for x, y, z, w in dataloader:
    print(x.shape, y.shape, z.shape, w.shape)
