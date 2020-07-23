# -*- coding: utf-8 -*-
"""This code :- it recognizes the face and ig two image given ,if both images are same
it gives true as output r else false.inadditional to , it gives the distance."""

import cv2
import face_recognition

imgbill=face_recognition.load_image_file("images/billgate_work.jpg")#importing image
imgbill=cv2.cvtColor(imgbill,cv2.COLOR_BGR2RGB) #converting BGR to RGB
imgbill_test=face_recognition.load_image_file("images/billgates_test.jpg")
imgbill_test=cv2.cvtColor(imgbill_test,cv2.COLOR_BGR2RGB)

faceloc=face_recognition.face_locations(imgbill)[0] #to detect the face location,it return 4 values
encodebill=face_recognition.face_encodings(imgbill)[0] #encoding image
#print(faceloc) 
cv2.rectangle(imgbill,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
                                                                      #color    #thickness

faceloc_test=face_recognition.face_locations(imgbill_test)[0]
encodebill_test=face_recognition.face_encodings(imgbill_test)[0]
#print(faceloc)
cv2.rectangle(imgbill_test,(faceloc_test[3],faceloc_test[0]),(faceloc_test[1],faceloc_test[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodebill],encodebill_test)#for comparing face encoding
facedis=face_recognition.face_distance([encodebill],encodebill_test)#to find distace b/w faces
#lower the distance ,better the matching of faces.
print(results,facedis) #result-bool facedis-value

cv2.putText(imgbill_test,f'{results}{round(facedis[0],2)}',(40,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,5),1)
                          #rounding to 2 decimal value    #origin               #font    #scale  #color #thickness
  

cv2.imshow("billgate",imgbill)#displaying image
cv2.imshow("billgate_test",imgbill_test)
cv2.waitKey(0)


