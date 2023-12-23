# -*- coding: utf-8 -*-
"""DL_HW7_BCE.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vOkfUCK9tRDyK4NwCPGCBz1fV60_Qyog
"""

pip install pytorch_fid

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import models,datasets,transforms
from PIL import Image
from tqdm import tqdm

import os
import matplotlib.pyplot as plt
import numpy as np

from pytorch_fid.fid_score import calculate_activation_statistics,calculate_frechet_distance
from pytorch_fid . inception import InceptionV3
import shutil

class IndexedDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path        
        #The transforms that will be applied to each image
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
          ])
        #Saving all the image locations
        self.image_filenames = []
        for (dirpath, dirnames, filenames) in os.walk(dir_path): 
            self.image_filenames += [os.path.join(dirpath, file) for file in filenames]
        
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx] #Getting the name of the image
        image = Image.open(img_name).convert('RGB') #opening the image
        image = self.transform(image) #Applying the transforms to the image
        return image

#Inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False)
        self.convt2 = nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False)
        self.convt3 = nn.ConvTranspose2d( 64 * 4, 64 * 2, 4, 2, 1, bias=False)
        self.convt4 = nn.ConvTranspose2d( 64 * 2, 64, 4, 2, 1, bias=False)
        self.convt5 = nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64 * 8)
        self.bn2 = nn.BatchNorm2d(64 * 4)
        self.bn3 = nn.BatchNorm2d(64*2)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        x = self.relu(self.bn1(self.convt1(input)))
        x = self.relu(self.bn2(self.convt2(x)))
        x = self.relu(self.bn3(self.convt3(x)))
        x = self.relu(self.bn4(self.convt4(x)))
        x = self.tanh(self.convt5(x))
        return x

#Inspired by https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64 * 2)
        self.bn3 = nn.BatchNorm2d(64 * 4) 
        self.bn4 = nn.BatchNorm2d(64 * 8)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.sigmoid(self.conv5(x))
        return x

#Initializing the weights
#From https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#Function to train the model
def training(epochs,optimizerD,optimizerG,criterion,netG,netD,train_data_loader,device,batch_size,model,m1,s1):
  #Storing the losses of the Generator and Discriminator
  train_lossesG = []
  train_lossesD = []
  fid_score = []

  #Assigning labels to the real and fake images
  real_label = 1
  fake_label = 0

  for epoch in range(epochs):
    running_lossG = 0.0
    running_lossD = 0.0
    for i,data in enumerate(train_data_loader):
      #Part 1: Training the Discriminator
      #Part A: On the real images
      optimizerD.zero_grad()
      inputs_real = data.to(device) #Getting the real image
      num_images = data.shape[0] #Getting the batch size
      labels_real = torch.full((num_images,),real_label,dtype=torch.float).to(device)  #Generating the labels for real images   
      
      outputs_real = netD(inputs_real).view(-1) #Getting the predicted labels for the real images
      loss_real = criterion(outputs_real, labels_real) #Calculating the loss on real images
      loss_real.backward()

      #Part B: On the fake images
      noise = torch.randn((num_images,100,1,1)).to(device) #Generating the loss
      inputs_fake = netG(noise) #Generating the fake images from this noise
      labels_fake = torch.full((num_images,),fake_label,dtype=torch.float).to(device) #Generating the labels for fake images
      outputs_fake = netD(inputs_fake.detach()).view(-1) #Getting the predicted labels for the fake images
      loss_fake = criterion(outputs_fake, labels_fake) #Calculating the loss on fake images
      loss_fake.backward()

      optimizerD.step()
      running_lossD += loss_real.cpu().item() + loss_fake.cpu().item()

      #Part 2: Training the Generator
      optimizerG.zero_grad()
      outputs_fake_G = netD(inputs_fake).view(-1) #Genrating the predicted labels for the fake images
      loss_G = criterion(outputs_fake_G, labels_real) #In the optimum case the fake images should be predicted as real by the discriminator
      loss_G.backward()
      optimizerG.step()
      running_lossG += loss_G.cpu().item()

    print("[epoch: %d, batch: %5d] Discriminator loss: %.3f Generator loss: %.3f" % (epoch + 1, i + 1, (running_lossD)/(i+1),(running_lossG)/(i+1)))
    train_lossesG.append(running_lossG/len(train_data_loader.dataset))
    train_lossesD.append(running_lossD/len(train_data_loader.dataset))
    fid = calc_fid(netG,device,m1,s1,model)
    fid_score.append(fid)
  return netG, train_lossesG, train_lossesD, fid_score

#Function to Plot a real image
def plot_image_real(dataset,index):
  img = dataset[index] #Getting the image
  i = np.transpose(np.asarray(img*127.5 + 127.5).astype(int),(1,2,0)) #converting the image from tensor to numpy
  ci = np.ascontiguousarray(i, dtype=np.uint8) #Making the array contiguous  
  return ci.astype(int)  #returns the image

#Function to Plot a fake image
def plot_image_fake(netG,device):
  noise = torch.randn((1,100,1,1)).to(device)
  img = netG(noise).cpu().detach() #Getting the image
  img = img.squeeze()
  i = np.transpose(np.asarray(img*127.5 + 127.5).astype(int),(1,2,0)) #converting the image from tensor to numpy
  ci = np.ascontiguousarray(i, dtype=np.uint8) #Making the array contiguous  
  return ci.astype(int)  #returns the image

#Function to save a fake image
def save_image_fake(img,i):
  im = Image.fromarray(img.astype(np.uint8))
  im.save("fake/"+str(i)+".jpeg")

#Function to calculate FID
def calc_fid(netG,device,m1,s1,model):
  #Generating 1000 fake images
  for i in range(1000):
    img = plot_image_fake(netG,device)
    save_image_fake(img,i)
  fake_paths = ["fake/" +str(i)+".jpeg" for i in range(0, 1000)] #Generating the path for the directory containing fake images
  #Obtaining the feature encoding for the real test images
  m2 , s2 = calculate_activation_statistics(fake_paths, model,device = device)
  fid_value = calculate_frechet_distance(m1,s1,m2,s2) #Calculating FID value
  print("FID: "+str(fid_value)) 
  return fid_value

#Function to Plot 4x4 images
def plot_collage(dataset,device,type_of="real",model=None):
  images = []
  if type_of=="real": #If you want real image generated
    indices = np.random.randint(0,len(dataset),16) #Getting 16 random indices
    for i in indices:
      images.append(plot_image_real(dataset,i)) #Saving the 16 images

  elif type_of=="fake": #If you want fake image generated
    for i in range(16):
      images.append(plot_image_fake(model,device)) #Saving the 16 images

  fig = plt.figure(figsize = (10,10))

  ax11 = fig.add_subplot(4,4,1)
  ax12 = fig.add_subplot(4,4,2)
  ax13 = fig.add_subplot(4,4,3)
  ax14 = fig.add_subplot(4,4,4)
  ax21 = fig.add_subplot(4,4,5)
  ax22 = fig.add_subplot(4,4,6)
  ax23 = fig.add_subplot(4,4,7)
  ax24 = fig.add_subplot(4,4,8)
  ax31 = fig.add_subplot(4,4,9)
  ax32 = fig.add_subplot(4,4,10)
  ax33 = fig.add_subplot(4,4,11)
  ax34 = fig.add_subplot(4,4,12)
  ax41 = fig.add_subplot(4,4,13)
  ax42 = fig.add_subplot(4,4,14)
  ax43 = fig.add_subplot(4,4,15)
  ax44 = fig.add_subplot(4,4,16)

  ax11.axis('off')
  ax12.axis('off')
  ax13.axis('off')
  ax14.axis('off')
  ax21.axis('off')
  ax22.axis('off')
  ax23.axis('off')
  ax24.axis('off')
  ax31.axis('off')
  ax32.axis('off')
  ax33.axis('off')
  ax34.axis('off')
  ax41.axis('off')
  ax42.axis('off')
  ax43.axis('off')
  ax44.axis('off')

  ax11.imshow(images[0])
  ax12.imshow(images[1])
  ax13.imshow(images[2])
  ax14.imshow(images[3])
  ax21.imshow(images[4])
  ax22.imshow(images[5])
  ax23.imshow(images[6])
  ax24.imshow(images[7])
  ax31.imshow(images[8])
  ax32.imshow(images[9])
  ax33.imshow(images[10])
  ax34.imshow(images[11])
  ax41.imshow(images[12])
  ax42.imshow(images[13])
  ax43.imshow(images[14])
  ax44.imshow(images[15])

train_dataset = IndexedDataset("/content/drive/MyDrive/pizzas/train")
test_dataset = IndexedDataset("/content/drive/MyDrive/pizzas/eval")

train_dataloader = DataLoader(train_dataset,batch_size=16,num_workers=64)
test_dataloader = DataLoader(test_dataset,batch_size=16,num_workers=64)

print(len(train_dataloader))
print(len(test_dataloader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

netG = Generator().to(device)
netG.apply(weights_init)
netD = Discriminator().to(device)
netD.apply(weights_init)

optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas = (0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0001, betas = (0.5, 0.999))
criterion = nn.BCELoss()
epochs = 40

#The number of parameters in the Generator
sum(p.numel() for p in netG.parameters() if p.requires_grad)

#The number of parameters in the Discriminator
sum(p.numel() for p in netD.parameters() if p.requires_grad)

#Creating an instance of the InceptionV3 model
dims = 2048
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model = InceptionV3([block_idx]).to(device)

#Obtaining the feature encoding for the real test images
real_paths = test_dataset.image_filenames
m1 , s1 = calculate_activation_statistics(real_paths,model,device = device)

trained_generator,train_lossesG, train_lossesD,fid_score = training(epochs,optimizerD,optimizerG,criterion,netG,netD,train_dataloader,device,16,model,m1,s1)

#Fake images
plot_collage(train_dataset,device,type_of="fake",model=trained_generator)

#Real Images
plot_collage(train_dataset,device,type_of="real",model=None)

plt.xlabel("Epochs")
plt.ylabel("FID value")
plt.title("FID value vs Epochs on the Test Dataset")
plt.plot(fid_score)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs on the Test Dataset") #This should be training dataset
plt.plot(train_lossesG,label="Generator")
plt.plot(train_lossesD, label="Discriminator")
plt.legend(loc = "upper right")

