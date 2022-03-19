import cv2
import numpy as np

def show(img, waitTime=0):
    # img[img > 255] = 255
    # img[img < 0] = 0
    cv2.imshow('output.png', np.uint8(img))
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()


def normalize(img):
    # img[img>255] = 255
    # img[img<0] = 0

    img = img - np.min(img)
    img = img / img.max() 
    img *= 255.9999 

    return img

def edgeFilter(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.float32([[1,2,1], [0,0,0],[-1,-2,-1]])
    Ix = cv2.filter2D(img1*1.0, -1, kernel)
    Iy = cv2.filter2D(img1*1.0, -1, kernel.T)
    I = (Ix**2+Iy**2)**.5
    # I = I.sum(axis=2) #cheap way to turn it into a gray image

    return I

def energyFilter(img):
    I = edgeFilter(img)
    erodeKernel = np.float32([[1, 1, 1]])
    h, w = img1.shape[:2]

    for j in range(h-1):
        rowAbove = I[j:j+1]
        y = cv2.erode(rowAbove, erodeKernel)
        I[j+1:j+2] += rowAbove
    # return np.pad(I,([0],[1]),constant_values=I.max()+1)

    return I

img1 = cv2.imread("Seam-Carving\\img1.jpg")
I = edgeFilter(img1)
# show(normalize(I))
energy = energyFilter(img1)
show(normalize(energy))

while 1:
    energy = energyFilter(img1)
    h, w = img1.shape[:2]
    out = img1*1
    y = h-1
    x = np.argmin(I[y]) #gets index that has the smallest x value in last row
    out[y, x:-1] = out[y, x+1:] #rethink where to put this
    while y:
        y -= 1
        x += np.argmin(energy[y, x-1:x+2]-1)
        out[y, x-1:-1] = out[y, x:]

    out = out[:, :-1]
    show(out)
    img1 = out



#FIND EDGES
# kernel = np.float32([[1,2,1], [0,0,0],[-1,-2,-1]])
# Ix = cv2.filter2D(img1*1.0, -1, kernel)
# Iy = cv2.filter2D(img1*1.0, -1, kernel.T)
# I = (Ix**2+Iy**2)**.5
# I = I.sum(axis=2)

# ORIGINAL TECHNIQUE
# show(I[::2, ::2])
# row = I[0]
# print(len(row))
# row = np.pad(row, (1, 1), 'edge')
# print(len(row))
# a = row[:-2]
# b = row[1:-1]
# c = row[2:]
# x = np.vstack((a, b, c))
# x = x.min(axis=0)
# #x and y should be the same thing, bc they do the same thing
# y = cv2.erode(I[0:1], np.float32([[1, 1, 1]]) ) 

# erodeKernel = np.float32([[1, 1, 1]])
# h, w = img1.shape[:2]

# for j in range(h-1):
#     rowAbove = I[j:j+1]
#     y = cv2.erode(rowAbove, erodeKernel)
#     I[j+1:j+2] += rowAbove
# I = np.pad(I, ([0], [1]), constant_values=I.max()+1)

#rotating image 90 degrees can do it vertically


#CARVING OUT THE SEAMS
# while 1:
#     energy = energyFilter(img1)
#     h, w = img1.shape[:2]
#     out = img1*1
#     y = h-1
#     x = np.argmin(I[y]) #gets index that has the smallest x value in last row
#     out[y, x:-1] = out[y, x+1:] #rethink where to put this
#     while y>=0:
#         y -= 1
#         x += np.argmin(energy[y, x-1:x+2]-1)
#         out[y, x-1:-1] = out[y, x:]

#     out = out[:, :-1]
#     show(out)
#     img1 = out




# out = normalize(I)
# show(out)
# x = np.argmin(I[-1])
# y = h-1
# #DRAWS SEAM
# while y:
#     out[y, x] = 255
#     out[y, x-1] = 255
#     out[y, x-2] = 255
#     x += np.argmin(I[y-1, x-1:x+2]-1)
#     y -= 1


