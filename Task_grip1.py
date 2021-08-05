#!/usr/bin/env python
# coding: utf-8

# #  TASK1
# 
# # Object detection using opencv
# 
# # The Sparks Foundation
# 
# # AUTHOR - VINAY DHIMAN

# In[1]:


#import Cv2 library

import cv2


# In[2]:


#import required YOLO files 

classNames=[]
classFile='data/coco.names'
configPath='data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weigthPath='data/frozen_inference_graph.pb'


# In[3]:


#On video cam for live stream

cap=cv2.VideoCapture(0)

with open(classFile) as f:
    classNames=f.read().rstrip('\n').split('\n')

net=cv2.dnn_DetectionModel(weigthPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success,img=cap.read()
    classIds,conf,bbox=net.detect(img,confThreshold=0.6)

    if len(classIds)!=0:
        for classId,confidence,box in zip(classIds.flatten(),conf.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.putText(img,str(round(confidence*100,2)), (box[0] +150, box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0,255), 2)


    cv2.imshow('image',img)
    cv2.waitKey(1)


# In[ ]:




