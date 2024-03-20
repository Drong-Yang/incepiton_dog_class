import numpy as np
import pandas as pd
import os
from PIL import Image



def luoData(path):
    lis = os.listdir('data1')
    print(lis)
    labels = []
    imgData = []
    dir_list = os.listdir("data1")
    label_list = [i for i in range(len(dir_list))]
    print(label_list)
    # 将数组转换成字典
    result_dict = {dir_list[i]: label_list[i] for i in range(len(dir_list))}
    for key, value in result_dict.items():
        sample = os.listdir(path + '/' + key)
        for i in sample:
            img = Image.open(path + "/" + key + '/' + i)
            img = img.convert("RGB")
            img = img.resize((64, 64), Image.ANTIALIAS)
            img = np.array(img)
            img = img / 255.0
            imgData.append(img)
            labels.append(value)
    imgData = np.array(imgData)
    labels = np.array(labels)

    return imgData, labels
