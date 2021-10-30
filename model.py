#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# To create your own model , download the dataset from given link
# save the downloaded folder in same directory as this project
# link: https://drive.google.com/drive/folders/1sr9ePH_OluMigAcd8XMZbFDyCEmQliid?usp=sharing


# In[ ]:





# In[ ]:


import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


# In[ ]:


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)


# In[ ]:


BS= 32
TS=(24,24)
train_batch= generator('dataset_new/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('dataset_new/test',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)


# In[ ]:


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
     Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
     Dropout(0.25),
     Flatten(),
     Dropout(0.5),
     Dense(4, activation='softmax')
])


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

compiled=model.fit(train_batch, validation_data=valid_batch,epochs=50,steps_per_epoch=SPE ,validation_steps=VS)


# In[ ]:


import pandas as pd

pd.DataFrame(compiled.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# In[ ]:


#To save the trained model
model.save('drowsiness_v_1.h5')


# In[ ]:


keras.backend.clear_session()

