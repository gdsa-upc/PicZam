# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as lt

import sys

img = cv2.imread ('C:/Users/Victor Pacheco Garci/Desktop/UPC/2017-18/GDSA/Team5/Farmacia Albinyana/farmacia_albinyana_101.jpg',1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT()

# busca directament els punts clau i els descriptors.
    # kp serà una llista de punts clau en una matriu numpy de Nombre de key points X128.
kp,des = sift.detectAndCompute(gray,None) 
lt.imshow (cv2.drawKeypoints(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),kp))
lt.show()