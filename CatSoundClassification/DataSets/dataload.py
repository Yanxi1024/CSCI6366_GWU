from torch.utils.data import Dataset, DataLoader, random_split
from pandas import read_csv
from numpy import load, pad

from numpy import load, pad



class SoundDataSet(Dataset):
    def __init__(self, csv_path, data_path, max_length=1600, num_channels=1):
        self.df = read_csv(csv_path, encoding='utf-8')
        self.data_path = data_path
        self.max_length = max_length
        self.num_channels = num_channels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df['label'][index]
        data = load(self.df['filepath'][index])

        # 确保数据为一维数组（单通道）
        if data.ndim > 1 and data.shape[1] != 0:
            data = data[:, 0]  # 取第一个通道的数据

        # 检查数据长度，并进行裁剪或填充
        if len(data) > self.max_length:
            data = data[:self.max_length]  # 裁剪多余的部分
        elif len(data) < self.max_length:
            data = pad(data, (0, self.max_length - len(data)), 'constant', constant_values=0)  # 使用0填充不足的部分

        # 转换数据类型为float32，并调整形状以适应卷积层
        data = data.astype('float32').reshape(1, self.max_length, 1)  # 将shape调整为 [channels, length, width]

        return data, label



def get_dataloader(data_dir, csv_path, batch_size, train_percent=0.9):
    dataset = SoundDataSet(data_dir, csv_path)
    num_sample = len(dataset)
    num_train = int(train_percent * num_sample)
    num_valid = num_sample - num_train
    train_ds, valid_ds = random_split(dataset, [num_train, num_valid])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                          persistent_workers=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
                          persistent_workers=True)
    return train_dl, valid_dl, len(dataset), len(train_ds), len(valid_ds)
