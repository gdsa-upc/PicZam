# -*- coding: utf-8 -*-
import cv2
import numpy as np

# leemos la imagen mediante la funcion imread
img = cv2.imread ('C:/Users/Victor Pacheco Garci/Desktop/UPC/2017-18/GDSA/Team5/Farmacia Albinyana/farmacia_albinyana_101.jpg',1)

# convertimos la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detectamos esquinas. Corner Harris(img,blockSize,ksize,k) utiliza los siguientes argumentos:
    # img - imagen en escala de grises tipo float 32.
    # blockSize - Tamaño del vecindario considerado para la detección de esquinas.
    # ksize - parámetro de apertura de la derivada de Sobel utilizada.
    # k - parámetro libre en la equación del detector Harris.
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.001)

# result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image. En este caso, dst>0.005
img[dst>0.005*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()