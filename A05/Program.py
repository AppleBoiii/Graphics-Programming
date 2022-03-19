import cv2
import numpy as np
from numpy.lib.npyio import save

def show(img, waitTime=0):
    if img.min()<0 or img.max()>255 or img.max() - img.min() < 20:
        img=normalize(img)
    cv2.imshow('output.png', np.uint8(img))
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def saveImg(img, name="output.png"):
    cv2.imwrite(name, img)
    print("Saved!")

def normalize(img):
    # img[img>255] = 255
    # img[img<0] = 0

    img = img - np.min(img)
    img = img / img.max() 
    img *= 255.9999 

    return img

def edgeFilter(img):
    grey = img
    if len(img.shape) > 2:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.float64([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = cv2.filter2D(grey*1.0, -1, kernel)
    Iy = cv2.filter2D(grey*1.0, -1, kernel.T)
    I = (Ix**2 + Iy**2)**0.5

    return I

def energyFilter(I):
    I = I*1.0
    kernel = np.float64([[1, 1, 1]]) #an erode kernel
    #eroding diminishes the features of an img / erodes boundaries of the foreground
    h, w = I.shape[:2]

    for j in range(h-1):
        row = I[j:j+1]
        row = cv2.erode(row, kernel)
        I[j+1:j+2] += row
    
    # return I
    return np.pad(I, ([0], [1]), constant_values=I.max()+1) #pads an extra pixel to each side 

def drawSeams(img, seams):
    out = img*1
    h = out.shape[0]
    for seam in seams:
        for i in range(h):
            y = i
            x = seam[i]
            try:
                out[y, x] = 255
            except:
                out[y, x-1] = 255
    
    show(out[::3, ::3])
    return out
    # saveImg(out, "Output\\40seams.png")

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

def delSeams(img, seams, view=False):
    out = img*1
    h = out.shape[0]
    for seam in seams:
        y = h-1
        while y:
            # y = i
            x = seam[y]
            if y == h-1:
                out[y, x:-1] = out[y, x+1:]
            else:
                out[y, x-1:-1] = out[y, x:]
            y -= 1
        out = out[:,:-1]
        if view:
            show(out, 30)
        # if seams.index(seam) % 1 == 0 and view:
        #     show(out, 33)
    return out

def resize(img, new_dim, view=False): #new_dim expected to be (wxh)
    img = img*1
    if len(new_dim) > 2:
        print("Not a dimension")
        return

    h, w = img.shape[:2]
    new_w, new_h = new_dim
    dw, dh = w-new_w, h-new_h

    if dw < 0 or dh < 0:
        print("Bad dimensions.")
        return
    
    energyMap = energyFilter(edgeFilter(img))
    # show(energyMap)
    vertical_seams = []
    for i in range(dw):
        energyMap = energyFilter(edgeFilter(img))
        seam = findSeam(img, energyMap)
        img = delSeams(img, [seam], view)
        vertical_seams.append(seam)
    
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    energyMap = energyFilter(edgeFilter(img))
    # show(energyMap)
    horizontal_seams = []
    for i in range(dh):
        energyMap = energyFilter(edgeFilter(img))
        seam = findSeam(img, energyMap)
        img = delSeams(img, [seam], view)
        horizontal_seams.append(seam)
    
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


orig_img = cv2.imread("Input\\landscapee.jpg")
img = orig_img*1
# print(img.shape)

#VERTICAL STUFF
edges = edgeFilter(img)
saveImg(edges, "vertical_edges.png")

energyMap = energyFilter(edgeFilter(orig_img))
show(energyMap[::3, ::3])
cv2.imwrite("vertical_energy.png", energyMap)

seam = findSeam(img, energyMap)
saveImg(drawSeams(img, [seam]), "vertical_seam_on_original.png")
saveImg(drawSeams(edges, [seam]), "vertical_seam_on_edges.png")
saveImg(drawSeams(energyMap, [seam]), "vertical_seam_on_energyMap.png")

no_seam = delSeams(img*1, [seam])
saveImg(no_seam, "vertical_seam_deleted.png")

seams = []
for i in range(100):
    seam = findSeam(img, energyMap)
    img = delSeams(img, [seam])
    energyMap = energyFilter(edgeFilter(img))
    seams.append(seam)

saveImg(drawSeams(orig_img, seams), "Output\\100_vertical_seams_drawn.png")
saveImg(img, "100_vertical_seams.png")






#HORIZONTAL STUFF
flipped = cv2.rotate(orig_img, cv2.ROTATE_90_CLOCKWISE)
show(flipped)

edges = edgeFilter(flipped)
show(edges[::2, ::2])
saveImg(cv2.rotate(edges, cv2.ROTATE_90_COUNTERCLOCKWISE), "horizontal_edges.png")

energyMap = energyFilter(edges)
temp = cv2.rotate(energyMap, cv2.ROTATE_90_COUNTERCLOCKWISE)
show(temp[::3, ::3])
saveImg(cv2.rotate(energyMap, cv2.ROTATE_90_COUNTERCLOCKWISE), "horizontal_energy.png")

seam = findSeam(flipped, energyMap)
t1 = drawSeams(flipped, [seam])
t2 = drawSeams(edges, [seam])
t3 = drawSeams(energyMap, [seam])
show(t3[::3, ::3])
show(cv2.rotate(t3, cv2.ROTATE_90_COUNTERCLOCKWISE))
saveImg(cv2.rotate(t1, cv2.ROTATE_90_COUNTERCLOCKWISE), "horizontal_seam_on_original.png")
saveImg(cv2.rotate(t2, cv2.ROTATE_90_COUNTERCLOCKWISE), "horizontal_seam_on_edges.png")
saveImg(cv2.rotate(t3, cv2.ROTATE_90_COUNTERCLOCKWISE), "horizontal_seam_on_energyMap.png")

no_seam = delSeams(flipped*1, [seam])
saveImg(cv2.rotate(no_seam, cv2.ROTATE_90_COUNTERCLOCKWISE), "horizontal_seam_deleted.png")

for i in range(100):
    seam = findSeam(flipped, energyMap)
    flipped = delSeams(flipped, [seam])
    energyMap = energyFilter(edgeFilter(flipped))
saveImg(cv2.rotate(flipped, cv2.ROTATE_90_COUNTERCLOCKWISE), "100_horizontal_seams.png")

print("DONE!!!!")
