import numpy as np
import cv2, math

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

img=cv2.imread("image.png")
h,w=img.shape[:2]
j=k=0
while True:
    for i in range(100):
		angle=math.pi*2*i/100+math.pi/2*np.cos(k*.05);
        scaleFactor=1+.5*np.cos(j*.1)
        T1=np.float64([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
        T2=np.float64([[1,0,w/2],[0,1,h/2],[0,0,1]])
        S=np.float64([[scaleFactor,0,0],[0,scaleFactor,0],[0,0,1]])
        R=np.float64([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        
        j+=1
        k+=1

        img2 = cv2.warpPerspective(img,M,(w,h))
        show(img,33,False)

