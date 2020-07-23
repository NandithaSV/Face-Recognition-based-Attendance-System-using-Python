# -*- coding: utf-8 -*-


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path="images"
image=[]
classnames=[]
mylist=os.listdir(path)
print(mylist)


#this loop imports all images from the path(images)
for cl in mylist:  
    curimg=cv2.imread(f'{path}/{cl}')#importing image
    image.append(curimg)
    classnames.append(os.path.splitext(cl)[0])#splitext() removes .jpg
print(classnames)


def findencodings(image): #to do encoding for all image
    encodelist=[]
    for img in image:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#converting image from BRG to RGB
        encode=face_recognition.face_encodings(img)[0] #encodings
        encodelist.append(encode)
    return (encodelist)


#recording attendance using name and time they arrived
#person attendace will record once ,when the person arrive @ first time.after recording if the same
#person comes infront of webcamp it won't get record
def markattendance(name):
    with open('attendance.csv','r+') as f: #.csv(comma seperated values) file used to display attendace
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0]) #entry[0] is name
        
        if name not in namelist:
            now=datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')
            
    
encodelistknown=findencodings(image)
print("encoding completed")

cap=cv2.VideoCapture(0) #for webcamp

while True:
    success,img=cap.read()
    imgs=cv2.resize(img,(0,0),None,0.25,0.25) #resizing the image(pixels,scale)    
    imgs=cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB) #converting image from BRG to RGB
    
    facecurframe=face_recognition.face_locations(imgs) #to detect the face location
    encodescurframe=face_recognition.face_encodings(imgs,facecurframe) #encoding


# the below for loop,one by one it will grab one face location from facecurframe list and it will grab 
#the encoding of encode frame from encodeframe.we want both in same loop so using zip()  
    for encodeface,faceloc in zip(encodescurframe,facecurframe):         
        matches=face_recognition.compare_faces(encodelistknown,encodeface)#comparing
        facedis=face_recognition.face_distance(encodelistknown,encodeface)#face distance
        print(facedis)
        matchindex=np.argmin(facedis)
        

#bounding box around face and writting thier respective         
        if matches[matchindex]:
            name=classnames[matchindex].upper() #converting names to uppercase
            print(name)
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4 #to review the actual value *4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)#drawing rectangle
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)#for display name
            markattendance(name)


    cv2.imshow('webcam',img)
    cv2.waitKey(1)



