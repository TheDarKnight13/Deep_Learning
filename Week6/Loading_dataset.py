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
                
    set_all = set() 
    for catId in catIds:
        category = class_list[coco_labels_inverse[catId]]
        imgId = coco.getImgIds(catIds=catId ) #Getting all the ids of the images of the category
        set_all = set_all.union(set(imgId))        
    
    imgIds = list(set_all) #contains all the image ids of images containing an object in class_list
    print("The total number of images is "+str(len(imgIds)))
    
    total_images = 0
    dictionary = {} #stores the annotations 
    for i in range(len(imgIds)):        
        img_url = coco.loadImgs(imgIds[i])[0] #url of the image
        img = io.imread(img_url['coco_url'])   #The actual image
        im = Image.fromarray(img) #Converting it to PIL object               
        im1 = im.resize((256,256))  #Resizing it
            
        annIds = coco.getAnnIds(img_url['id'],iscrowd=False) #Gets the annotation ids
        anns = coco.loadAnns(annIds) #gets the actual annotations  
        array_info = []
        for k in range(len(anns)):         
            if(anns[k]["area"]>4096 and anns[k]["category_id"] in catIds): #Checking if area of the bounding box > 4096                
                #Resizing the bounding box coordintaes
                [x1,y1,w1,h1] = anns[k]["bbox"]
                x  = x1*256.0/im.size[0]
                w  = w1*256.0/im.size[0]
                y  = y1*256.0/im.size[1]
                h  = h1*256.0/im.size[1] 
                cl = class_list[coco_labels_inverse[anns[k]["category_id"]]] 
                array_info.append([[x,y,w,h],cl])                                
            else:
                continue
                
        if total_images==0 and len(array_info)>0: #Creating the directories
            dataset = os.path.join("Dataset",dtype) 
            os.makedirs(dataset)
        
        if len(array_info)>0: #If the image contains atleast one object of area 4096 belonging to one of the 3 categories, its saved
            im1.save(os.path.join(dataset,str(total_images)+".jpeg")) #Saving the image
            dictionary[os.path.join(dataset,str(total_images)+".jpeg")] = array_info #Saving bounding box coordinates and class
            total_images = total_images+1

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


# In[26]:


#Number of training images having "n" annotations
a = {}
for i in range(15):
    a[i] = 0
for loc in train_dict:
    a[len(train_dict[loc])] +=1
a


# In[28]:


#Number of testing images having "n" annotations
b = {}
for i in range(14):
    b[i] = 0
for loc in test_dict:
    b[len(test_dict[loc])] +=1
b


# In[ ]:




