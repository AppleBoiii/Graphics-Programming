import numpy as np
import cv2

def show(img, waitTime=0):
    if img.min() < 0 or img.max() > 255 or img.max() - img.min() < 20:
        img=normalize(img)
    cv2.imshow('image', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def normalize(img):
    img=img-np.min(img)
    print("NORMALIZING!!!")
    img/=img.max()
    img*=255.99999
    return np.uint8(img)

def edgeFilter(img):
    temp=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel=np.float32([[1,2,1],[0,0,0],[-1,-2,-1]])
    Ix=cv2.filter2D(temp*1.0,-1,kernel)
    Iy=cv2.filter2D(temp*1.0,-1,kernel.T)
    I=(Ix**2+Iy**2)**.5
    return I
    
def energyFilter(img):
    edge=edgeFilter(img)
    # edge[717:808,110:145]=-255
    h,w=edge.shape[:2]
    errosionKernel=np.float32([[1,1,1]])
    for j in range(h-1):
        temp=edge[j:j+1]
        temp = cv2.erode(temp,errosionKernel )
        edge[j+1:j+2]+=temp
    return np.pad(edge,([0],[1]),constant_values=edge.max()+1)

def findSeam(img, energyMap):
    seam = []
    h = img.shape[0]

    y = h-1 #starts at the bottom row
    x = np.argmin(energyMap[y])

    seam.append(x)
    while y:
        y -= 1
        x += np.argmin(energyMap[y, x-1:x+2])-1
        seam.append(x)
    
    # energyMap = delSeams(energyMap, [seam])

    return seam[::-1]    

img=cv2.imread("Seward SeamCarving\\image.jpeg")

# show(energyFilter(img))

energy=energyFilter(img)
seam = findSeam(img, energy)



# while 1:
#     energy=energyFilter(img)
#     # ~ show(energy[::2,::2],1)
#     h,w=img.shape[::2]
#     out=img*1
#     y=h-1
#     x=np.argmin(energy[y])
#     out[y,x:-1]=out[y,x+1:]
#     while y:
#         y-=1
#         x+=np.argmin(energy[y,x-1:x+2])-1
#         out[y,x-1:-1]=out[y,x:]
#     out=out[:,:-1]
#     show(out[::2,::2],1)
#     img=out

# img=cv2.imread("image.jpeg", 0) #grayscale
# kernel = np.float64([[1, 2, 1], [0, 0, 0], [-1,-2, -1]])
# Ix = cv2.filter2D(img*1.0, -1, kernel)#if img isn't 64 bit, it might not show correctly
# Ix = np.abs(Ix)
# Ix = Ix > 50

# Iy = cv2.filter2D(img*1.0, -1, kernel.T)#if img isn't 64 bit, it might not show correctly
# Iy = np.abs(Iy)
# Iy = Iy > 50

# corners = Ix & Iy
# show(np.uint8(corners)*255)


 


# DRAW SEAM
# ~ out=normalize(I)
# ~ show(normalize(I)[::2,::2])
# ~ x=np.argmin(I[-1])
# ~ y=h-1
# ~ while y:
    # ~ out[y,x]=255
    # ~ out[y,x-1]=255
    # ~ out[y,x-2]=255
    # ~ x+=np.argmin(I[y-1,x-1:x+2])-1
    # ~ y-=1
# ~ show(out[::2,::2])


