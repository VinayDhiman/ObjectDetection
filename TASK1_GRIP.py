#!/usr/bin/env python
# coding: utf-8

# #  THE SPARKS FOUNDATION
# 
# ## TASK 1
# 
# ## OBJECT DETECTION USING OPENCV
# 
# ## AUTHOR - VINAY DHIMAN
# 
# 
# 
# 

# In[5]:


# import libraries
import cv2
import matplotlib.pyplot as plt


# In[6]:


# importing ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt and frozen_inference_graph.pb

config_file = 'data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

frozen_model = 'data/frozen_inference_graph.pb'

file_name = 'data/coco.names'


# In[7]:


# test pretrained model

classLabels = [0]

with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    classLabels.append(fpt.read())
    
test = cv2.dnn_DetectionModel(frozen_model, config_file)
test.setInputSize(550, 320)
test.setInputScale(1.5 / 127.5)  
test.setInputMean((127.5, 127.5, 127.5)) 
test.setInputSwapRB(True)


# In[8]:


# importing "traffic video.mp4"
cap = cv2.VideoCapture("data/walk.mp4")  

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise print("cannot open video")

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    ClassIndex, confidence, bbox = test.detect(frame, confThreshold=0.55)

    #print(ClassIndex)
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0), thickness=2)

    cv2.imshow('Live Object Detection', frame)

    if cv2.waitKey(2) & 0XFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




