#!/usr/bin/env python
# coding: utf-8

# In[3]:


#Dowloading the embeddings
import gensim.downloader as gen_api
from gensim.models import KeyedVectors 


# In[5]:


word_vectors = gen_api.load("word2vec-google-news-300")               
word_vectors.save( 'vectors.kv')


# In[ ]:




