import cv2
import matplotlib.pyplot as plt

img = cv2.imread ('C:/Users/Victor Pacheco Garci/Desktop/UPC/2017-18/GDSA/Team5/Farmacia Albinyana/farmacia_albinyana_101.jpg',1)

#Crea l'objecte surf
#Configurem Hessian Threshold a 400
surf = cv2.SURF(10)

#calcula els punts clau i verifica el seu numero
kp, des = surf.detectAndCompute(img,None)
len(kp)

#verifica l'umbral actual de Hesse
#ek En casos reals, es millor tenir valors entre 300 i 500
surf.hessianThreshold = 400
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),0)
plt.imshow(img2),plt.show()