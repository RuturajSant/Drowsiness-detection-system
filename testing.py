#!/usr/bin/env python
# coding: utf-8

# In[56]:


import cv2
import os
from keras.models import load_model
import numpy as np


# In[57]:


model = load_model('models/100_Epochs_trained_model_new.h5')


# In[58]:


labels = {'Closed': 0, 'Open': 1, 'no_yawn': 2, 'yawn': 3}


# In[61]:


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),0)
        cv2.imshow("image",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if img is not None:
            images.append(img)
    return images

images = load_images_from_folder('test')


# In[62]:


#resizing images to fit our model
for image in images:
    image = cv2.resize(image,(24,24))
    image = image/255
    image = image.reshape(24,24,-1)
    image = np.expand_dims(image,axis=0)
    preds = model.predict(image)
    predlabel=[]
    for mem in preds:
        predlabel.append(np.argmax(mem))
        key_list=list(labels.keys())
        for i in predlabel:
            print(key_list[i])


# In[8]:





# In[9]:





# In[10]:





# In[ ]:




