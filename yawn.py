#!/usr/bin/env python
# coding: utf-8

# In[6]:


import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time


# In[7]:


mixer.init()
sound = mixer.Sound('alarm.wav')


# In[8]:


face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')


# In[9]:


lbl=['Yawn','Okay']

model = load_model('models/drowsiness_v_1.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]


# In[10]:


while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        isGood = False
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
        f_frame = frame[y:y+h,x:x+w]
        count = count+1
        f_frame = cv2.cvtColor(f_frame,cv2.COLOR_BGR2GRAY)
        f_frame = cv2.resize(f_frame,(24,24))
        f_frame = f_frame/255
        f_frame = f_frame.reshape(24,24,-1)
        f_frame = np.expand_dims(f_frame,axis=0)
        fpred = model.predict(f_frame)
        fpred = np.argmax(fpred,axis=-1)
        if(fpred[0]==3):
            lbl='Yawn'
            isGood = True
        if(fpred[0]==2):
            lbl='Okay'
            isGood = False
        break

    if(fpred[0]==3 and isGood):
        score = score+1
        cv2.putText(frame,"Yawn",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Okay",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    if(score>5):#change this to increase/decrease threshold for alarm to play.
        #play alarm
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:
            pass
        if(thicc<12):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




