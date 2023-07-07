from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Dataset_My(Dataset):
    def __init__(self, path=r".\data\ETTm1.csv", size=(384, 96), target=["OT"], flag="train", scale=True):
        super(Dataset_My, self).__init__()
        
        self.target = target
        self.seq = size[0]
        self.pred = size[1]
        
        if flag == "train":
            self.stage = 0
        elif flag == "val":
            self.stage = 1
        else:
            self.stage = 2

        board1s = [0, 360 * 96 - self.seq, 480 * 96 - self.seq]
        board2s = [360 * 96, 480 * 96, 600 * 96]
        board1 = board1s[self.stage]
        board2 = board2s[self.stage]

        data_pd = pd.read_csv(path)
        data = data_pd[data_pd.columns[1:]]
        self.data = data[board1:board2]
        self.data_x = self.data.values
        self.data_y = self.data[self.target].values

        self.scaler = StandardScaler()
        if scale:
            train_data = data[board1s[0]:board2s[0]].values
            self.scaler.fit(train_data)
            self.data_x = self.scaler.transform(self.data_x)
        
        stamp = data_pd[['date']][board1:board2]
        stamp['date'] = stamp['date'].map(pd.to_datetime)
        stamp['month'] = stamp['date'].map(lambda x:x.month)
        stamp['weekday'] = stamp['date'].map(lambda x:x.weekday())
        stamp['day'] = stamp['date'].map(lambda x:x.day)
        stamp['hour'] = stamp['date'].map(lambda x:x.hour)
        stamp['minute'] = stamp['date'].map(lambda x:x.minute // 15)
        stamp = stamp.drop(labels='date', axis=1)
        self.stamp = stamp.values


    def __getitem__(self, index):
        sx = index
        ex = index + self.seq
        sy = ex
        ey = sy + self.pred
        x = self.data_x[sx:ex]
        y = self.data_y[sy:ey]
        xm = self.stamp[sx:ex]
        ym = self.stamp[sy:ey]
        return x, xm, y, ym


    def __len__(self):
        return len(self.data) - self.pred - self.seq + 1

if __name__ == "__main__":
    dataset = Dataset_My()
    for i in range(len(dataset)):
        x = dataset[i]
        print(i, x[0].shape, x[1].shape, x[2].shape, x[3].shape)
