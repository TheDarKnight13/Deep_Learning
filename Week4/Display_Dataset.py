#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# In[2]:


fig = plt.figure(figsize = (15,6))

ax11 = fig.add_subplot(3,5,1)
ax12 = fig.add_subplot(3,5,6)
ax13 = fig.add_subplot(3,5,11)

ax11.axis('off')
ax12.axis('off')
ax13.axis('off')

ax11.imshow(Image.open('Dataset/train/airplane/1.jpeg'))
ax12.imshow(Image.open('Dataset/train/airplane/2.jpeg'))
ax13.imshow(Image.open('Dataset/train/airplane/3.jpeg'))

ax21 = fig.add_subplot(3,5,2)
ax22 = fig.add_subplot(3,5,7)
ax23 = fig.add_subplot(3,5,12)

ax21.axis('off')
ax22.axis('off')
ax23.axis('off')

ax21.imshow(Image.open('Dataset/train/bus/1.jpeg'))
ax22.imshow(Image.open('Dataset/train/bus/2.jpeg'))
ax23.imshow(Image.open('Dataset/train/bus/3.jpeg'))

ax31 = fig.add_subplot(3,5,3)
ax32 = fig.add_subplot(3,5,8)
ax33 = fig.add_subplot(3,5,13)

ax31.axis('off')
ax32.axis('off')
ax33.axis('off')

ax31.imshow(Image.open('Dataset/train/cat/1.jpeg'))
ax32.imshow(Image.open('Dataset/train/cat/2.jpeg'))
ax33.imshow(Image.open('Dataset/train/cat/3.jpeg'))

ax41 = fig.add_subplot(3,5,4)
ax42 = fig.add_subplot(3,5,9)
ax43 = fig.add_subplot(3,5,14)

ax41.axis('off')
ax42.axis('off')
ax43.axis('off')

ax41.imshow(Image.open('Dataset/train/dog/1.jpeg'))
ax42.imshow(Image.open('Dataset/train/dog/2.jpeg'))
ax43.imshow(Image.open('Dataset/train/dog/3.jpeg'))

ax51 = fig.add_subplot(3,5,5)
ax52 = fig.add_subplot(3,5,10)
ax53 = fig.add_subplot(3,5,15)

ax51.axis('off')
ax52.axis('off')
ax53.axis('off')

ax51.imshow(Image.open('Dataset/train/pizza/1.jpeg'))
ax52.imshow(Image.open('Dataset/train/pizza/2.jpeg'))
ax53.imshow(Image.open('Dataset/train/pizza/3.jpeg'))


# In[ ]:




