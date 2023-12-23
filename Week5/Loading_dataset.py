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
import cv2
import json


# In[2]:


#Function to download the dataset
def download_dataset(class_list,coco,dtype):
    catIds = coco.getCatIds(class_list) #Gets the ids for the pizza, bus and cat classes
    categories = coco.loadCats(catIds) #Gets the id,category,supercategory name for the id
    categories.sort(key = lambda x:x['id']) #sorts the hashmap in the order of ids
    
    coco_labels_inverse = {} #Stores a hashmap where the key is the category id and the value is the index of the category in the class_list
    for idx, in_class in enumerate(class_list):
        for c in categories:
            if c['name'] == in_class:
                coco_labels_inverse[c['id']] = idx
                
    dictionary = {} #stores the annotations  
    for catId in catIds:
        category = class_list[coco_labels_inverse[catId]]
        print(category)   #Prints the category    
        imgIds = coco.getImgIds(catIds=catId ) #Getting all the ids of the images of the category
        total_images=0
        print("The total number of images is "+str(len(imgIds)))
        for i in range(len(imgIds)):        
            img_url = coco.loadImgs(imgIds[i])[0] #url of the image
            img = io.imread(img_url['coco_url'])   #The actual image
            im = Image.fromarray(img) #Converting it to PIL object               
            im1 = im.resize((256,256))  #Resizing it
            
            annIds = coco.getAnnIds(img_url['id'],catIds=catId,iscrowd=False) #Gets the annotation ids
            anns = coco.loadAnns(annIds) #gets the actual annotations
                        
            for k in range(len(anns)):
                if(anns[k]["area"]>40000 and anns[k]["category_id"]==catId): #Checking if the area of the bounding box is greater than 40,000
                    #Resizing the bounding box coordintaes
                    [x1,y1,w1,h1] = anns[k]["bbox"]
                    x  = x1*256.0/im.size[0]
                    w  = w1*256.0/im.size[0]
                    y  = y1*256.0/im.size[1]
                    h  = h1*256.0/im.size[1]                 
                                
                    if total_images==0: #Creating the directories
                        dataset = os.path.join("Dataset",dtype,category) 
                        os.makedirs(dataset)               
                    im1.save(os.path.join(dataset,str(total_images)+".jpeg")) #Saving the image
                    dictionary[os.path.join(dataset,str(total_images)+".jpeg")] = [x,y,w,h] #Saving its annotations
                    
                    total_images = total_images+1
                    break
                    
                else:
                    continue
        print("The total number of images with a dominant object is "+str(total_images))
    return dictionary


# In[3]:


coco1=COCO("instances_train2014.json")
coco2=COCO("instances_val2014.json")
class_list = ["pizza","bus","cat"]


# In[4]:


#For the training dataset
train_dict = download_dataset(class_list,coco1,"train")
json1 = json.dumps(train_dict) #Creating a JSON object
f = open("dict_train.json","w") #opening file for writing
f.write(json1) #writing 
f.close() #closing


# In[5]:


#For the validation dataset
test_dict = download_dataset(class_list,coco2,"test")
json2 = json.dumps(test_dict) #Creating a JSON object
f = open("dict_test.json","w") #opening file for writing
f.write(json2) #writing 
f.close() #closing


# In[6]:


#Number of images in the training dataset
len(train_dict)


# In[7]:


#Number of images in the validation dataset
len(test_dict)

