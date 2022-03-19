import numpy as np
import cv2
import math
# https://en.wikipedia.org/wiki/Kernel_(image_processing)

def show(img, waitTime=0):
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

def apply(img, kernel):
    return cv2.filter2D(img, -1, kernel)

def boxBlur(img, n=3, save=False):
    kernel = np.ones((n, n), dtype=np.float64) / n**2
    out = apply(img, kernel) #borderType default is reflect101

    if save:
        saveImg(out, "Output\\boxblur1.png")

    # show(out)
    return out

def boxBlur2(img, n=3, save=False): #figure this out
    kernel1 = (np.ones((n, 1), dtype=np.float64)) / n #n x 1
    kernel2 = kernel1.T                       #1 x n
    
    print(kernel2*kernel1)
    #save each of these
    out1 = apply(img, kernel1)
    out2 = apply(out1, kernel2)
    # out2 = np.uint8(out2)

    orig = boxBlur(img, n)
    diff = orig*1.0 - out2*1.0
    x = np.average(abs(diff))
    # print(f"The diff is +{diff}")
    print(x) #save this

    diff = normalize(diff)
    show(diff)

    if save:
        saveImg(out2, "Output\\boxblur2.png")
        saveImg(diff, "Output\\boxblurdiff.png")
    

def nCr(n, k):
    f = math.factorial #saves the function to variable r
    return f(n) / f(k) / f(n-k) #plugs in the values for the function stored in r

def pascalsTriangle(n):
    n = n-1
    triangle = [1]
    for k in range(1, n):
        triangle.append(nCr(n, k))
    triangle.append(1)

    return np.array(triangle, dtype=np.float64)
    
def gaussianBlur(img, n=3, save=False):
    kernel = np.zeros((n, n), dtype=np.float64)
    for row in range(len(kernel[0])):
        triangle = pascalsTriangle(n)
        kernel[row] = triangle*int(triangle[row])
    kernel /= kernel.sum()

    out = apply(img, kernel)

    if save:
        saveImg(out, "Output\\gaussianblur1.png")

    show(out)
    return out

def gaussianBlur2(img, n=3, save=False):
    kernel1 = np.zeros((n, 1), dtype=np.float64)
    kernel1 = pascalsTriangle(n)
    kernel1 /= kernel1.sum()
    kernel2 = kernel1.T

    out1 = apply((img), kernel1)
    out2 = apply(out1, kernel2)
    out2 = normalize(out2)

    orig = gaussianBlur(img, n)
    diff = orig - out2
    x = np.average(abs(diff))
    print(f"The diff is +{diff}")
    print(x) #save this

    diff = normalize(diff)
    show(diff)

    if save:
        saveImg(out2, "Output\\\gaussianblur2.png")
        saveImg(diff, "Output\\gaussianblurdiff.png")

def edgeFilter(img, save=False): #do bonus
    # kernel = np.array(([-1, 0, 1],[-2, 0, 2], [-1, 0, 1]), dtype=np.float64) #Gx
    # kernel = np.array(([-1, -2, -1],[0, 0, 0], [1, 2, 1]), dtype=np.float64) #Gy
    kernel = np.array(([0, 1, 2], \
                        [-1, 0, 1], \
                        [-2, -1, 0]))
    #diagnol edges

    out = cv2.filter2D(img, -1, kernel)
    show(out)
    if save:
        saveImg(out, "Output\\edgeFilter.png")
    return out

def sharpen(img, save=False):
    # kernel = np.array(([0, -1, 0], [-1, 5, -1], [0, -1, 0]))
    kernel = np.array(([0, 0, -1, 0, 0], \
      [0, -1, -2, -1, 0], \
      [-1, -2, 17, -2, -1], \
      [0, -1, -2, -1, 0], \
      [0, 0, -1, 0, 0], ))
    out = cv2.filter2D(img, -1, kernel)

    show(normalize(out))
    if save:
        saveImg(out, "Output\\sharpen.png")
    return out

def corners(img):
    out = img*1
    #combine horizontal and vertical kernel?
    kernel1 = np.array(([-1, 0, 1], \
                        [-2, 0, 2], \
                        [-1, 0, 1]))

    kernel2 = np.array(([-1, -2, -1], \
                        [0, 0, 0], \
                        [1, 2, 1]))

    out1 = cv2.filter2D(img, -1, kernel1)
    out2 = cv2.filter2D(img, -1, kernel2)


    out1[(out1>0) == (out2>0)] = 255
    out1[(out1>0) != (out2>0)] = 0
    show(out)
    return out 

    '''
    theoretically, if you can find the horizontal and vertical edges, then finding corners is a step above that. I would use
    two sobel filters to do this. The places where the vertical and horizontal lines intersect should be corners. Find the points of intersection,
    subtract off the rest of the horizontal and vertical areas, and then the resulting image should be the corners.
    '''

img1 = cv2.imread("Input\\image1.jpg")
img3 = cv2.imread("Input\\image3.jpg")
img4 = cv2.imread("Input\\img4.png")
show(img1)


# show(img1)
# boxBlur(img1, 5, save=True)
boxBlur2((img1), n=5, save=True) #work on this
# gaussianBlur(np.float64(img1),  save=True)
# gaussianBlur2(np.float64(img1))
# edgeFilter((img4)) #get bonus on this
# sharpen((img1), save=True) #get bonus on this
# corners(img4) 
