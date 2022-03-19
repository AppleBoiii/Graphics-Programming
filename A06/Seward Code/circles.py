# ~ Find an label the circles in an images
# ~ Find and outline something with an irregular shape.
# ~ 90% synthetic images with one size
# ~ * multiple sizes, real images, rotations, more parameters etc

# ~ patch=np.zeros((w,h,sd,rd))
# ~ for s in scale divisions:
    # ~ for r in rotation divisions:
        # ~ patch[:,:,s,r]=resized rotate version of source
        # ~ patch[:,:,s,r]/=patch[:,:,s,r].sum()







import numpy as np
import cv2
import sys

def show(img, waitTime=0,forceNormalization=False):
    if forceNormalization or img.min()<0 or img.max()>255:
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
def toBase(x,b):
    digits="0123456789abcdefghijklmnopqrstuvwxyz"
    output=""
    while x:
        d=x%b
        output=digits[d]+output
        x//=b
    return output


def edgeFilter(img):
    temp=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel=np.float32([[1,2,1],[0,0,0],[-1,-2,-1]])
    Ix=cv2.filter2D(temp*1.0,-1,kernel)
    Iy=cv2.filter2D(temp*1.0,-1,kernel.T)
    I=np.hypot(Ix,Iy)
    return I
    
def energyFilter(img):
    edge=edgeFilter(img)
    h,w=edge.shape[:2]
    errosionKernel=np.float32([[1,1,1]])
    for j in range(h-1):
        temp = edge[j]
        temp = cv2.erode(temp[:,None],errosionKernel )[0]
        edge[j+1]+=temp
    return np.pad(edge,([0],[1]),constant_values=edge.max()+1)
    
# ~ print(toBase(1024234,10))
# ~ print(toBase(1024234,34))
# ~ print(int("q20i",38))
# ~ sys.exit(9)

img=cv2.imread("Seward Code\\circles.webp")[::4,::4]*1
# img[::30]=0
# img[:,::30]=0
# show(img)

h,w=img.shape[:2]
# ~ edges=edgeFilter(img)
# ~ show(edges)
edges=cv2.Canny(img,50,20)
# ~ show(img)
# ~ show(edges)
d=min(h,w)


Y,X,D=np.mgrid[:d,:d,:d]
dist=np.hypot(Y-d//2,X-d//2)
cone=((D>3)*(dist>D/2-1)*(dist<D/2+1))/(D+.1)
cone=cone[:,:,::-1]
print(img.shape,cone.shape)
# ~ circle=np.zeros((d,d),dtype=np.uint8)
# ~ cv2.circle(circle,(d//2,d//2),d//2,255)
# ~ show(circle)
import scipy.signal

votes=scipy.signal.correlate(edges[:,:,None]*1.0,cone*1.0)
show(votes[:,:,4])
# ~ Y,X=np.where(edges)
# ~ i=0
# ~ for x,y in zip(X,Y):
    
    # ~ votes[y:y+d,x:x+d]+=circle
    # ~ print(i*100/len(X))
    # ~ i+=1

h,w,d=votes.shape

# ~ i=0
# ~ while 1:
    # ~ slice=votes[i%h]
    # ~ show(normalize(slice).T,33)
    # ~ i+=1


# ~ y,x,d2 = np.unravel_index(np.argmax(votes), votes.shape)
# ~ print(votes.shape)
# ~ print(x,y,d2)
# ~ cv2.circle(img,(x-d//2,y-d//2),d2//2,(0,255,0),2)
import scipy.ndimage.morphology
# ~ votes=normalize(votes)
votes.max()
dilated_votes=scipy.ndimage.morphology.grey_dilation(votes, (5,5,5))
peaks=((dilated_votes==votes)*(votes>votes.max()*.7))
print(peaks.sum())
show(peaks[:,:,20]*255.0)
y,x,d2=np.where(peaks)
for i,j,r in zip(x,y,d2//2):
    print(votes[j,i,r*2])
    cv2.circle(img,(i-d//2,j-d//2),r,(0,255,0),2)
show(img)

# ~ img[...][patch>0]=255
