#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch 
import torch.nn as nn
from torchvision import datasets,models,transforms,ops
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
import time
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import seaborn as sn


# In[2]:


def convert_to_bbox_cord(yolo_tensor_aug):
  num_cells_along_width = 8
  yolo_interval = 32
  #Keeping track of index for which anchor box is best for all 64 cells
  best_ancbox_for_cell = {cx : None for cx in range(64)}
  for cx in range(yolo_tensor_aug.shape[0]):
    curr_best = 0
    cell_tensor = yolo_tensor_aug[cx] #Shape (5x9)
    #Finding the anchor box index for which the object has highest probability of being present in a cell
    for ax in range(cell_tensor.shape[0]):
      if cell_tensor[ax][0] > cell_tensor[curr_best][0]: 
        curr_best = ax    
    best_ancbox_for_cell[cx] = curr_best

  #Sorting this dictionary in descending order of probability of a cell having an object
  sorted_icx = sorted(best_ancbox_for_cell,key=lambda x: yolo_tensor_aug[x,best_ancbox_for_cell[x]][0].item(), reverse=True)
  retained_cells_idxs = sorted_icx[:5]

  #Extracting the labels and bounding boxes from the top 5 cells
  labels = []
  bbox = []
  for cx in retained_cells_idxs:
    yolo_vector = yolo_tensor_aug[cx,best_ancbox_for_cell[cx],:] #Extracting the yolo vector
    class_label_prob_all = nn.Softmax(dim=0)(yolo_vector[5:]) #Applying softmax to the class labels
    class_label_prob = class_label_prob_all[:-1]
    if yolo_vector[0]<0.25:
      continue
    else:
      #Extracting the label
      labels.append(torch.argmax(class_label_prob).item())

      #Extracting the height and width of the bounding box
      h = yolo_vector[3].item()*yolo_interval
      w = yolo_vector[4].item()*yolo_interval

      #Extracting the center coordinate of the bounding box wrt the cell
      del_x = yolo_vector[1].item()
      del_y = yolo_vector[2].item()

      #Getting the cell coordinates
      cell_row_idx = cx//num_cells_along_width
      cell_col_idx = cx%num_cells_along_width

      #Finding the yolo cell center coordinates
      cell_center_i = (cell_row_idx*yolo_interval + float(yolo_interval)) / 2.0 
      cell_center_j = (cell_col_idx*yolo_interval + float(yolo_interval)) / 2.0 

      #Finding the center coordinates of the bounding box
      bb_height_center = del_x*yolo_interval + cell_center_j  
      bb_width_center =  del_y*yolo_interval + cell_center_i

      #Finding top left and right of the bounding box
      x =  bb_width_center - w/2.0
      y =  bb_height_center - h/2.0 
      bbox.append([x,y,x+w,y+h])
  
  return torch.Tensor(bbox),torch.Tensor(labels)                                                                


# In[3]:


class IndexedDataset(Dataset):

    def __init__(self, dir_path):
        self.dir_path = dir_path
        
        if os.path.basename(self.dir_path) == 'train': #transforms for the train dataset
          self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
          ])
          f = open('Dataset/dict_train.json')
          self.bbox_data = json.load(f)
          f.close()
          self.loc = "Dataset\\train\\"  
          
        elif os.path.basename(self.dir_path) == 'test': #transforms for the test dataset
          self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
          ])
          f = open('Dataset/dict_test.json')
          self.bbox_data = json.load(f)
          f.close()
          self.loc = "Dataset\\test\\"
        
        image_filenames = []
        for (dirpath, dirnames, filenames) in os.walk(dir_path): #Saving all the image locations
            image_filenames += [os.path.join(dirpath, file) for file in filenames]
        self.image_filenames = image_filenames    
        self.labels_map = {"bus" : 0, "cat": 1, "pizza" : 2} #Creating hashmap of the class and a number

    def __len__(self):
        return len(self.image_filenames)

    def convert_to_yolo_tensor(self,bbox,label):
      #bbox is of shape (number_of_objects x 4)
      #label is of shape (number_of_objects)
  
      yolo_interval = 32
      cell_height = yolo_interval
      cell_width = yolo_interval
      num_cells_along_width = 8
      num_cells_along_height = 8
      yolo_tensor = torch.zeros(64,5,8)
      yolo_tensor_aug = torch.zeros(64,5,9)

      for ix in range(len(bbox)):
        #Center coordinates of the bounding box  
        bb_height_center = (bbox[ix,1] + bbox[ix,3])/2.0 
        bb_width_center = (bbox[ix,0] + bbox[ix,2])/2.0

        #height and width bounding box 
        obj_bb_height = bbox[ix,3] - bbox[ix,1]
        obj_bb_width = bbox[ix,2] - bbox[ix,0]
          
        #Finding the index of the yolo cell
        cell_row_indx = int(bb_height_center / yolo_interval)
        cell_col_indx = int(bb_width_center / yolo_interval)
           
        #Center coordinates of the yolo cell
        cell_center_i = (cell_row_indx*yolo_interval + float(yolo_interval)) / 2.0 
        cell_center_j = (cell_col_indx*yolo_interval + float(yolo_interval)) / 2.0 

        #Center coordinates of the bounding box wrt to the yolo cell
        del_x = (bb_height_center - cell_center_j) / yolo_interval 
        del_y = (bb_width_center - cell_center_i) / yolo_interval

        #Height and width of the bounding box wrt to the yolo cell
        bh = obj_bb_height/ yolo_interval 
        bw = obj_bb_width/ yolo_interval

        #Finding the correct anchor box
        ratio = obj_bb_height/obj_bb_width
        if ratio <= 0.2: 
          anc_box_i = 0 
        elif 0.2 < ratio <= 0.5: 
          anc_box_i = 1
        elif 0.5 < ratio <= 1.5: 
          anc_box_i = 2 
        elif 1.5 < ratio <= 4.0: 
          anc_box_i = 3 
        elif ratio > 4.0: 
          anc_box_i = 4

        #Constructing the yolo vector
        yolo_vector = torch.FloatTensor([1,del_x.item(), del_y.item(), bh.item(), bw.item(), 0, 0, 0] ) 
        yolo_vector[5 + int(label[ix].item())] = 1

        #Inserting the yolo vector into the yolo tensor 
        cell_index = cell_row_indx * num_cells_along_width + cell_col_indx
        yolo_tensor[cell_index, anc_box_i] = yolo_vector
        
      ## If no object is present, throw all the probability mass into the extra 9th element of yolo vector
      yolo_tensor_aug[:,:,:-1] = yolo_tensor[:,:,:]
      for cx in range(64): # Over all the yolo cells
        for ax in range(5): #  Over all the anchor boxes
          if yolo_tensor_aug[cx,ax,0] == 0: 
            yolo_tensor_aug[cx,ax,-1] = 1

      return yolo_tensor_aug  

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        bbox_loc  = self.loc + os.path.basename(img_name)
        bbox = self.bbox_data[bbox_loc]
        cl = []
        bb = []
        for i in range(len(bbox)):
          cl.append(self.labels_map[bbox[i][1]])
          b = bbox[i][0]
          bb.append(b)    
        #x and y have traditional math notations
        #Changing the last two variables from w,h to x2,y2
        bb = torch.Tensor(bb)
        bb[:,2] = bb[:,2]+bb[:,0]
        bb[:,3] = bb[:,3]+bb[:,1]
        
        yolo_tensor = self.convert_to_yolo_tensor(bb,torch.Tensor(cl))

        return image, yolo_tensor


# In[4]:


#Function to Plot the image with the ground truth annotation
def plot_image_ground(model,dataset,index,device):
  img = dataset[index][0] #Getting the image
  i = np.transpose(np.asarray(img*127.5 + 127.5).astype(int),(1,2,0)) #converting the image from tensor to numpy
  ci = np.ascontiguousarray(i, dtype=np.uint8) #Making the array contiguous
  labels_map = {0:"bus", 1: "cat", 2: "pizza"}

  #Getting the ground truth bounding box and labels
  yolo_tensor = dataset[index][1] 
  bbox, labels = convert_to_bbox_cord(yolo_tensor)

  #Plotting the ground truth bounding boxes
  for i in range(len(labels)):
    cl = labels_map[int(labels[i].item())] #Class of the image
    [x1,y1,x2,y2] = bbox[i] #Getting the ground truth bounding box coordinates     
    ci = cv2.rectangle(ci,(int(x1),int(y1)),(int(x2),int(y2)),(36,255,12),2) #drawing the ground truth bounding box
    ci = cv2.putText(ci,cl,(int(x1),int(y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(36,255,12),2)

  #Getting the predicted bounding boxes
  pred_yolo_tensor = model(img.unsqueeze(dim=0).to(device)).squeeze()
  pbbox, plabels = convert_to_bbox_cord(pred_yolo_tensor.cpu())

  #Plotting the predicted bounding boxes
  for i in range(len(plabels)):
    pcl = labels_map[int(plabels[i].item())] #Class of the image
    [x1,y1,x2,y2] = pbbox[i] #Getting the predicted bounding box coordinates     
    ci = cv2.rectangle(ci,(int(x1),int(y1)),(int(x2),int(y2)),(255,36,12),2) #drawing the predicted bounding box
    ci = cv2.putText(ci,cl,(int(x1),int(y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,36,12),2)

  
  plt.imshow(ci)  #plotting the image with the bounding box
  plt.axis("off")
  


# In[5]:


class ResnetBlock(nn.Module):
  def __init__(self,input_nc,output_nc,stride):
    super().__init__()
    self.conv1  = nn.Conv2d(input_nc,output_nc,kernel_size=3,stride=stride,padding=1)
    self.bn1 = nn.BatchNorm2d(output_nc)
    self.relu1  = nn.ReLU(True)
    self.conv2  = nn.Conv2d(output_nc,output_nc,kernel_size=3,stride=1,padding=1)
    self.bn2 = nn.BatchNorm2d(output_nc)
    self.conv3  = nn.Conv2d(input_nc,output_nc,kernel_size=3,stride=stride,padding=1)
    self.bn3 = nn.BatchNorm2d(output_nc)
    self.relu3  = nn.ReLU(True)
    self.down = True if (stride!=1 or input_nc!=output_nc) else False

  def forward(self,x):
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    
    if self.down:
      identity = self.conv3(identity)
      identity = self.bn3(identity)

    x = x+identity
    x = self.relu3(x)

    return x


# In[6]:


class HW6Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
    self.bn1  = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(True)
    self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1)
    self.block1 = ResnetBlock(64,64,1)
    self.block2 = ResnetBlock(64,64,1)
    self.block3 = ResnetBlock(64,128,2)
    self.block4 = ResnetBlock(128,128,1)
    self.block5 = ResnetBlock(128,256,2)
    self.block6 = ResnetBlock(256,256,1)
    self.block7 = ResnetBlock(256,512,2)
    self.block8 = ResnetBlock(512,512,1)
    self.fc = nn.Linear(512*8*8,8*8*5*9)

  def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    xc = self.block1(x)
    xc = self.block2(xc)
    xc = self.block3(xc)
    xc = self.block4(xc)
    xc = self.block5(xc)
    xc = self.block6(xc)
    xc = self.block7(xc)
    xc = self.block8(xc)
    xc = xc.view(xc.shape[0],-1)
    xc = self.fc(xc)
    xc = xc.view(xc.shape[0],64,5,9)
    return xc



    


# In[7]:


def calculate_loss(data_loader,criterion1,criterion2,criterion3,net,train,optimizer):
    running_loss = 0.0
    running_loss1 = 0.0
    running_loss2 = 0.0
    running_loss3 = 0.0
    for i, data in enumerate(data_loader):
      inputs, yolo_tensor_aug = data
      inputs = inputs.to(device)
      yolo_tensor_aug = yolo_tensor_aug.to(device) 
      
      #Passing the image through the model to get the predicted yolo tensor
      if train:
        optimizer.zero_grad()
      predictions_aug = net(inputs)

      #Calculating the losses
      loss = torch.tensor(0.0, requires_grad=True).float().to(device) #Overall loss

      #Only going ahead with the yolo vectors where there is an object predicted or actually present 
      idx1  = torch.where(predictions_aug.view(-1,9)[:,0]>0)[0]
      idx2  = torch.where(yolo_tensor_aug.view(-1,9)[:,0]>0.5)[0]
      idx = list(set(torch.Tensor.tolist(idx1)).union(set(torch.Tensor.tolist(idx2))))
      
      if len(idx) == 0:
        continue

      #All the ground truth and predicted yolo vectors that satisfy the above condition
      pred_yolo_vector = predictions_aug.view(-1,9)[idx] 
      target_yolo_vector = yolo_tensor_aug.view(-1,9)[idx]

      #Loss for the presence of the object (BCE)
      pred_obj = nn.Sigmoid()(pred_yolo_vector[:,0])
      target_obj = target_yolo_vector[:,0] 
      loss1 = criterion1(pred_obj,target_obj)
            
      #Loss for the bounding box coordinates (MSE)
      pred_cor = pred_yolo_vector[:,1:5]
      target_cor = target_yolo_vector[:,1:5]
      loss2 = criterion2(pred_cor,target_cor)
            
      #Loss for the class of the object (CE)
      pred_cl = pred_yolo_vector[:,5:]
      target_cl = torch.max(target_yolo_vector[:,5:].data, 1)[1]
      loss3 = criterion3(pred_cl,target_cl)

      loss = loss1 + loss2 + loss3
      if train:             
        loss.backward()
        optimizer.step()
        
      running_loss += loss.cpu().item()
      running_loss1 += loss1.cpu().item()
      running_loss2 += loss2.cpu().item()
      running_loss3 += loss3.cpu().item()

    return net,running_loss, running_loss1, running_loss2, running_loss3


# In[8]:


def training(epochs,optimizer,criterion1,criterion2,criterion3,net,train_data_loader,test_data_loader,device,scheduler):
  train_losses = []
  train_losses1 = []
  train_losses2 = []
  train_losses3 = []

  test_losses = []
  test_losses1 = []
  test_losses2 = []
  test_losses3 = []

  for epoch in range(epochs):
    print("epoch: "+str(epoch))
    net.train()
    net,running_loss, running_loss1, running_loss2, running_loss3 = calculate_loss(train_data_loader,criterion1,criterion2,criterion3,net,True,optimizer)         
    train_losses.append(running_loss/len(train_data_loader))
    train_losses1.append(running_loss1/len(train_data_loader))
    train_losses2.append(running_loss2/len(train_data_loader))
    train_losses3.append(running_loss3/len(train_data_loader))
    print("[Training] total loss: %.3f BCE loss: %.3f MSE loss: %.3f CE loss: %.3f" % (train_losses[-1],train_losses1[-1],train_losses2[-1],train_losses3[-1]))

    net.eval()
    net,test_loss, test_loss1, test_loss2, test_loss3 = calculate_loss(test_data_loader,criterion1,criterion2,criterion3,net,False,optimizer)         
    test_losses.append(running_loss/len(test_data_loader))
    test_losses1.append(running_loss1/len(test_data_loader))
    test_losses2.append(running_loss2/len(test_data_loader))
    test_losses3.append(running_loss3/len(test_data_loader))
    print("[Testing] total loss: %.3f BCE loss: %.3f MSE loss: %.3f CE loss: %.3f" % (test_losses[-1],test_losses1[-1],test_losses2[-1],test_losses3[-1]))

    scheduler.step()
    
  return net, [train_losses,train_losses1,train_losses2,train_losses3], [test_losses,test_losses1,test_losses2,test_losses3]


# In[9]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[10]:


train_dataset = IndexedDataset("/content/drive/MyDrive/Dataset_HW6/train")
test_dataset = IndexedDataset("/content/drive/MyDrive/Dataset_HW6/test")
train_dataloader = DataLoader(train_dataset,batch_size=96,shuffle=True,num_workers=96)
test_dataloader = DataLoader(test_dataset,batch_size=96,shuffle=True,num_workers=96)
print(len(train_dataset))
print(len(test_dataset))


# In[11]:


model = HW6Net()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, betas = (0.9,0.99))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)

criterion1 = nn.BCELoss()
criterion2 = nn.MSELoss()
criterion3 = nn.CrossEntropyLoss()
epochs = 100


# In[12]:


sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[13]:


trained_model,train_losses,test_losses = training(epochs,optimizer,criterion1,criterion2,criterion3,model,train_dataloader,test_dataloader,device,scheduler)


# In[18]:


plot_image_ground(trained_model,test_dataset,2,device)


# In[23]:


plot_image_ground(trained_model,test_dataset, 3,device)


# In[45]:


plot_image_ground(trained_model,test_dataset, 25,device)


# In[51]:


plot_image_ground(trained_model,test_dataset, 30,device)


# In[55]:


plot_image_ground(trained_model,test_dataset, 34,device)


# In[57]:


plot_image_ground(trained_model,test_dataset, 37,device)


# In[66]:


plot_image_ground(trained_model,test_dataset, 47,device)


# In[71]:


plot_image_ground(trained_model,test_dataset, 52,device)


# In[78]:


plot_image_ground(trained_model,test_dataset, 59,device)


# In[89]:


plot_image_ground(trained_model,test_dataset, 70,device)


# In[90]:


plot_image_ground(trained_model,test_dataset, 71,device)


# In[92]:


plot_image_ground(trained_model,test_dataset, 73,device)


# In[100]:


plot_image_ground(trained_model,test_dataset, 81,device)


# In[113]:


plot_image_ground(trained_model,test_dataset, 92,device)


# In[132]:


plot_image_ground(trained_model,test_dataset, 111,device)


# In[162]:


plot_image_ground(trained_model,test_dataset, 140,device)


# In[171]:


plot_image_ground(trained_model,test_dataset, 148,device)


# In[185]:


plot_image_ground(trained_model,test_dataset, 162,device)


# In[202]:


plot_image_ground(trained_model,test_dataset, 178,device)


# In[205]:


plot_image_ground(trained_model,test_dataset, 181,device)


# In[216]:


plot_image_ground(trained_model,test_dataset, 192,device)


# In[217]:


plot_image_ground(trained_model,test_dataset, 193,device)


# In[236]:


#TRAIN DATASET
plot_image_ground(trained_model,train_dataset, 11,device)


# In[239]:


#TRAIN DATASET
plot_image_ground(trained_model,train_dataset, 14,device)


# In[244]:


#TRAIN DATASET
plot_image_ground(trained_model,train_dataset, 19,device)


# In[249]:


#TRAIN DATASET
plot_image_ground(trained_model,train_dataset, 24,device)


# In[250]:


#TRAIN DATASET
plot_image_ground(trained_model,train_dataset, 25,device)


# In[251]:


#TRAIN DATASET
plot_image_ground(trained_model,train_dataset, 26,device)


# In[255]:


#TRAIN DATASET
plot_image_ground(trained_model,train_dataset, 30,device)


# In[263]:


#TRAIN DATASET
plot_image_ground(trained_model,train_dataset, 38,device)


# In[268]:


#TRAIN DATASET
plot_image_ground(trained_model,train_dataset, 43,device)


# In[271]:


#Plotting the training loss vs epochs for all the three models
epochs = np.arange(1,101)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs epochs")
plt.plot(epochs,train_losses[0],label="Total")
plt.plot(epochs,train_losses[1],label="BCE")
plt.plot(epochs,train_losses[2],label="MSE")
plt.plot(epochs,train_losses[3],label="CE")
plt.legend(loc = "upper right")


# In[272]:


#Plotting the testing loss vs epochs for all the three models
epochs = np.arange(1,101)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Testing Loss vs epochs")
plt.plot(epochs,test_losses[0],label="Total")
plt.plot(epochs,test_losses[1],label="BCE")
plt.plot(epochs,test_losses[2],label="MSE")
plt.plot(epochs,test_losses[3],label="CE")
plt.legend(loc = "upper right")


# In[ ]:




