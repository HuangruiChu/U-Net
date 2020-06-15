# encoding: utf-8
"""
@author: Ming Cheng
@contact: ming.cheng@dukekunshan.edu.cn
"""

import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


transform_train = T.Compose([   
        T.Resize([256, 256]),
        T.RandomRotation(degrees=45, expand=True),
        T.CenterCrop([224,224]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        

transform_test = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  
        

class MyDataset(data.Dataset):
    
    
    def __init__(self, file_path=None, classes=None, transform=None):   
        """
        初始化自定义Dataset类的参数
        Attributes
            file_path: 字符串，数据集的存储路径，例如‘./UCF101/train’ 或 './UCF101/eval'等
            classes  : 列表，每个元素为一个字符串，代表一个子类别，例如['dog', 'airplane', ...]等
            transform: 传入一个从torchvision.transforms定义的数据预处理
        """
        self.classes = classes
        self.transform = transform
        # 初始化给定文件夹下的所有数据
        self.init_all_data(file_path) 

        return None
        

    def init_all_data(self, file_path):
        """
        初始化该数据集内所有的图像及其对应的标签，保存在self.images和self.labels两个列表内
        Attributes
            file_path: 字符串，数据集文件夹的存储路径
        """
        # 初始化两个列表，记录该数据集内每一张图片的完整路径及其对应的标签
        self.images = []
        self.labels = []
        # 遍历所有的子类别，并得到每个子类别对应的文件夹路径
        for idx, cls in enumerate(self.classes):
            cls_path = os.path.join(file_path, cls)
            # 遍历当前子类文件夹下的所有图片
            for img in os.listdir(cls_path):
                # 得当当前图片的完整路径，若是有效图片，则记录该图片的完整路径及其标签
                img = os.path.join(cls_path, img)
                if self.is_valid_image(img):
                    self.images.append(img)
                    self.labels.append(idx)

        return None

        
    def is_valid_image(self, img_path):
        """
        判断图片是否为可以打开的有效文件
        Attributes
            img_path: 字符串，待检测图片的存储路径
        Returns
            valid: 布尔变量，True/False分别表示该图片是否可以正常打开
        """
        try:
            # 若读取成功，设valid为True
            i = Image.open(img_path)
            valid = True
        except:
            # 若读取失败，设valid为False
            valid = False
            
        return valid
        

    def __getitem__(self, idx):
        """
        按给定索引，获取对应的图片及其标签
        Attributes
            idx: int类型数字，表示目标图像的索引
        Returns
            image: 一个打开的PIL.Image对象，是PIL库存储图像的一种数据格式（类似于OpenCV利用numpy张量存储图像）
            label: int类型，表示对应的类别，例如假设self.classes=['cat', 'dog', 'airplane']，则label=1代表‘dog'类别；
                   对于pytorch的分类，不需要特意将其变成onehot向量，因为crossentropy函数内置了这部分功能。
        """
        # 利用PIL.Image.open打开图片，并将其强制转化为RGB格式（防止数据集中混杂灰度图，导致读取出单通道图片，送入网络因矩阵维度不一致而报错）
        image = Image.open(self.images[idx]).convert('RGB')
        # 获取对应的标签
        label = self.labels[idx]
        # 进行预处理的变换
        if self.transform:
            image = self.transform(image)
              
        return image, label
   

    def __len__(self):
        """
        获取数据集中图像的总数，该方法的作用是用于DataLoader去调用，从而获取在给定Batch Size的情况下，一个Epoch的总长，
        从而可以在一个Epoch结束时实现shuffle数据集的功能
        """

        return len(self.images)