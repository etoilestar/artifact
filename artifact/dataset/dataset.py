import os
# import cv2
import pydicom
from torch.utils.data import DataLoader, Dataset
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import albumentations #推荐的数据增强库，https://blog.csdn.net/qq_27039891/article/details/100795846
#还有两个备选数据增强库：torchvision中的transformer和imgaug

class MYdataloader(Dataset):
    def __init__(self, path, mode='train', debug=False, load_method='train'):#定义训练/验证数据的相关信息，包括路径，数据增强等，形成数据的列表
        super(MYdataloader, self).__init__()
        self.mode = mode
        self.path = path
        self.load_method = load_method
        self.datalist = sorted(glob(os.path.join(path, '*/*/*/DS_CORCTA_0_75*20-80*/*.IMA')))
        self.labellist = sorted(glob(os.path.join(path, '*/*/*/DS_CORCTA_0_75*BEST*/*.IMA')))
        assert len(self.datalist) == len(self.labellist), "data number and label number not match"
        l = self.__len__()
        if mode == 'train':
            self.datalist = self.datalist[:int(0.9*l)]
            self.labellist = self.labellist[:int(0.9 * l)]
        else:
            self.datalist = self.datalist[int(0.9*l):]
            self.labellist = self.labellist[int(0.9 * l):]
        if debug:
            self.datalist = self.datalist[:10]
            self.labellist = self.labellist[:10]

    def __getitem__(self, index):#分batch的读取数据，通过index指向获得__init__中得到的数据的位置
        data_file = self.datalist[index]
        data = pydicom.read_file(data_file)
        data_array = data.pixel_array #shape=(512,512)
        data_array = np.expand_dims(data_array, axis=0)

        label_file = self.labellist[index]
        label = pydicom.read_file(label_file)
        label_array = label.pixel_array
        label_array = np.expand_dims(label_array, axis=0)

        if self.load_method == 'view':
            plt.imshow(data_array,cmap='gray')
            plt.title('before')
            plt.show()
            plt.imshow(label_array,cmap='gray')
            plt.title('later')
            plt.show()
        return data_array/255.0, label_array/255.0 #返回的数据为numpy或者PIL的类型均可

    def __len__(self):#返回数据的长度
        return len(self.datalist)

if __name__ == '__main__':
    dataloader = DataLoader(MYdataloader(path='G:\CT_DATA',load_method='view'), batch_size=1, shuffle=False,num_workers=1)#这一步定义了dataloader，走到了__init__那步
    for step, (data,label) in enumerate(dataloader):#这一步开始在getitem中读取并处理数据
        pass
