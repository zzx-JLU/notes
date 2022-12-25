from torch.utils.data import Dataset
import os
from PIL import Image


# 建立 Dataset 的子类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.files = os.listdir(self.path)

    # 重写 __getitem__() 方法
    def __getitem__(self, index):
        file_name = self.files[index]
        file_path = os.path.join(self.path, file_name)
        img = Image.open(file_path)
        label = self.label_dir
        return img, label

    # 重写 __len__() 方法
    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    train_dir = './data/hymenoptera_data/train/'
    ants_dir = 'ants'
    bees_dir = 'bees'

    ants_data = MyData(train_dir, ants_dir)
    img, label = ants_data[0]
    img.show()

    bees_data = MyData(train_dir, bees_dir)
