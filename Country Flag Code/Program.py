import cv2
import numpy as np
import math
#do south korea flag
WIDTH = 640
HEIGHT = 480
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)
ORANGE = (0, 128, 255)
RED = (61, 27, 141)

def star(img, color, cx, cy, r):
    h, w = img.shape[:2]

    y, x = np.mgrid[:h, :w]
    cx = WIDTH/2
    cy = HEIGHT/2
    starRadius = 50
    numPoints = 5
    numLoops = 2
    sectorAngle = np.pi*2/numPoints
    #convert to polar coordinates
    theta = (np.arctan2(y-cy, x-cx)) # + np.pi) / 2 / np.pi*255
    r = np.hypot(y-cy, x-cx)

    #five is the number of points
    theta %= sectorAngle #makes five triangles / places where it goes black to white

    #reflect about middle of sector
    theta = np.minimum(theta, sectorAngle-theta)
    tipAngle = np.pi-2*np.pi*numLoops/numPoints
    m = -np.tan(tipAngle/2)

    #convert to cartesian coordiantes
    # theta += np.pi/2
    xp = r*np.cos(theta)
    yp = r*np.sin(theta)

    #color under line
    img[yp < m*(xp-starRadius )] = 0

def normalize(img):
    img = img - np.min(img) #makes sure the lowest value is 0 in the image
    img /= img.max() #now you get values between 0 and 1
    img *= 255.9999 

    return np.uint8(img)


def show(img):
    # h, w = img.shape[:2] #gets shape of first two dimensions
    # if h>640:
    #     pass
    # elif w > 480:
    #     pass

    cv2.imshow('output.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows

img = np.zeros((HEIGHT, WIDTH, 3), dtype = np.uint8)

# img[:HEIGHT//2] = CYAN #NUMPY thing: fills the top half of the image to this color 
# img[:, :WIDTH//2] = ORANGE
img[:] = WHITE

y, x = np.mgrid[:HEIGHT, :WIDTH] #gives hxw array of all x and y coordinates

# QARAR FLAG
unitSize = 40
rowHeight = 11*unitSize/18
rowPolarity = (y//rowHeight)%2 > 0
cond1 = y%rowHeight<11/36*(x-8*unitSize)
cond2 = rowHeight - y%rowHeight<11/36*(x-8*unitSize)
img[cond2*rowPolarity] = RED
img[cond1*np.invert(rowPolarity)] = RED

#CIRCLES
circle = (x-WIDTH/2)**2+(y-HEIGHT/2)**2<100**2
img[circle*(img[:, :,2] == 141)] = 255
img[:] = 255
#STAR
cx = WIDTH/2
cy = HEIGHT/2
starRadius = 50
numPoints = 5
numLoops = 1
sectorAngle = np.pi*2/numPoints
#convert to polar coordinates
theta = (np.arctan2(y-cy, x-cx)) # + np.pi) / 2 / np.pi*255
# print(theta)
r = np.hypot(y-cy, x-cx)
# print(r)

#five is the number of points
theta %= sectorAngle #makes five triangles / places where it goes black to white

#reflect about middle of sector
theta = np.minimum(theta, sectorAngle-theta)
tipAngle = np.pi-2*np.pi*numLoops/numPoints
# print(tipAngle)
m = -np.tan(tipAngle/2)
# print(m)

#convert to cartesian coordiantes
# theta += np.pi/2
xp = r*np.cos(theta)
yp = r*np.sin(theta)

#color under line
img[yp < m*(xp-starRadius )] = 0

# show(theta)
# show(normalize(theta))


# img[(x-400)**2+(y-400)**2<100**2] = 255 draws a white circle


cv2.imwrite('output.png', img)
print("Done.")
show(img)

'''
cv2.imshow('output.png', img[::2, ::2]) #subsamples and skips every other pixel

cv2.waitKey(0)
cv2.destroyAllWindows
'''