
import numpy as np
import pandas as pd


import cv2
import os
#/truba/home/muozturk/1/*.jpg
DATADIR='/truba/home/muozturk/'
CATEGORIES=['1','2']

IMG_SIZE=32

training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR, category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_data() 

import random
random.shuffle(training_data)

X=[]
y=[]

for categories, label in training_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE)
print(X.shape)
print(len(y))
np.save("/truba/home/muozturk/train.npy",X)
np.save("/truba/home/muozturk/trainLabel.npy",y)

