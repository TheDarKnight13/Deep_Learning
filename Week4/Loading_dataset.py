#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Referred to the COCO API Github repo
#importing the libraries
from pycocotools.coco import COCO
import os
import time
import random
import requests 
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from PIL import Image


# In[2]:


coco=COCO("annotations_trainval2014/annotations/instances_train2014.json")


# In[3]:


#Function to download the dataset
def download_dataset(categories):
    for category in categories:
        print(category)
        catIds = coco.getCatIds(catNms=[category]) #Getting the id of the particular id
        imgIds = coco.getImgIds(catIds=catIds ) #Getting all the ids of the images of the category
        img_idxs = random.sample(imgIds, 2200) #getting the image indices of 2000 images
        total_images = 0
        for i in range(len(img_idxs)):
            if total_images == 2000:
                break
            img_url = coco.loadImgs(img_idxs[i])[0] #url of the image
            img = io.imread(img_url['coco_url'])   #The actual image
            im = Image.fromarray(img) #Converting it to PIL object
            if im is None: #To make sure that no NoneType objects are in the dataset
                continue
            im1 = im.resize((64,64))  #Resizing it
            if total_images==0: #Creating the directories
                train = os.path.join("Dataset","train",category)
                test = os.path.join("Dataset", "test",category)
                os.makedirs(train)
                os.makedirs(test)
            if total_images<1500: #Saving first 1500 images to the training dataset
                im1.save(os.path.join(train,str(i)+".jpeg"))
            elif total_images>=1500: #Saving next 500 images to the testing dataset
                im1.save(os.path.join(test,str(i)+".jpeg"))
            total_images = total_images +1
        print(i)


# In[4]:


categories = ['airplane', 'bus', 'cat', 'dog', 'pizza']
download_dataset(categories)


# In[ ]:




