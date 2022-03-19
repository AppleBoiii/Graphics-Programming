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
    img = img*1.0
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
    edge[717:808,110:145]=-255
    h,w=edge.shape[:2]
    errosionKernel=np.float32([[1,1,1]])
    for j in range(h-1):
        temp=edge[j:j+1]
        temp = cv2.erode(temp,errosionKernel )
        edge[j+1:j+2]+=temp
    return np.pad(edge,([0],[1]),constant_values=edge.max()+1)
    

img = cv2.imread("Seward Code\\img.png")
h, w = img.shape[:2]
edges = cv2.Canny(img, 50, 20) #shows all edges evenly
d = 400 #diameter of cirlce
f = 20
votes = np.zeros(((h+d), (w+d), d//f), dtype=np.uint32)

# makes manual circle
Y, X, D = np.mgrid[:d, :d, :d//f]
# circle = np.hypot(Y-d//2, X-d//2) == d//2
dist = np.hypot(Y-d//2, X-d//2)
circle = (dist>D*f/2-1)*(dist<D*f/2+1)

# makes circle automatically
# circle = np.zeros(((d), (d)), dtype=np.uint8)
cv2.circle(circle, (d//2, d//2), d//2, 255)
# show(circle)

Y, X = np.where(edges) #gives Y and Xs for where there are edges
for x, y in zip(X, Y): #goes through the first elements of each array
    votes[y:y+d, x:x+d] += circle
# votes = normalize(votes)
h, w, d = votes.shape
for i in range(w):
    slice = votes[i]
    show(normalize(slice), 33)
# show(normalize(votes[:, :, 200//f]))