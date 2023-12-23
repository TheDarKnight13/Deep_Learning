#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math as m


# In[2]:


class Sequence():
    def __init__(self,array):
        self.array = array
        
    def __len__(self):
        return len(self.array)
        
    def __iter__(self):
        return Sequence_iter(self)
    
    def __gt__(self,other):
        if len(self.array)!=len(other.array):
            raise ValueError("Two arrays are not equal in length!")
        else:
            count = 0
            for i in range(len(self.array)):
                if self.array[i]>other.array[i]:
                    count = count+1
            
            return count
        
class Sequence_iter():
    def __init__(self,obj):
        self.items = obj.array
        self.index = -1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index= self.index+1
        if self.index<len(self.items):
            return self.items[self.index]
        else:
            raise StopIteration        


# In[3]:


class Fibonacci(Sequence):
    def __init__(self,first_value,second_value):
        super().__init__([])
        self.first_value = first_value
        self.second_value = second_value
        
    def series(self):
        t1 = self.first_value
        t2 = self.second_value
        self.array = [t1,t2]
        for i in range(self.length-2):
            t3 = t2
            t2 = t1+t2
            t1 = t3
            self.array.append(t2)                    
        
    def __call__(self,length):
        self.length = length
        self.array =[]
        self.series()
        print(self.array)
        
        


# In[4]:


class Prime(Sequence):
    def __init__(self):
        super().__init__([])
    
    def series(self):
        i = 2
        while(len(self.array)<self.length):
            flag=0;
            for j in range(2,int(m.sqrt(i))+1):
                if i%j==0:
                    flag = 1
                    break            
            if flag == 0:
                self.array.append(i)
            i = i+1
        
    def __call__(self,length):
        self.length = length
        self.array =[]
        self.series()
        print(self.array)
    


# #### Task 3

# In[5]:


FS = Fibonacci(first_value=0,second_value=1)
FS(6)


# #### Task 4

# In[6]:


print(len(FS))


# In[7]:


print([n for n in FS]) 


# #### Task 5 

# In[8]:


#Task 5
PS = Prime()
PS(7)


# In[9]:


print(len(PS))


# In[10]:


print([n for n in PS]) 


# #### Task 6

# In[11]:


FS = Fibonacci(first_value=0,second_value=1)
FS(12)


# In[12]:


PS = Prime()
PS(12)


# In[13]:


print(FS>PS)


# In[14]:


PS(11)


# In[15]:


print(FS>PS)


# In[ ]:




