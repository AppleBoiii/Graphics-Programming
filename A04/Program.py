import cv2
import numpy as np
from sklearn.cluster import KMeans

def show(img, waitTime=0,forceNormalization=False):
    if forceNormalization or img.min()<0 or img.max()>255:
        img=normalize(img)
    cv2.imshow('image', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def normalize(img):
    img=img-np.min(img)
    img = img / img.max()
    img*=255.99999
    return np.uint8(img)

def goodMatches(matches, N):
    goodMatches = []
    for match1, match2 in matches:
        if match1.distance < N*match2.distance:
            goodMatches.append(match1)
    
    return goodMatches

def getHomography(kp1, kp2, goodMatches):
    srcPoints = []
    dstPoints = []

    for match in goodMatches:
        index1 = match.queryIdx
        index2 = match.trainIdx

        srcPoint = kp1[index1].pt
        dstPoint = kp2[index2].pt

        srcPoints.append([srcPoint])
        dstPoints.append([dstPoint])


    srcPoints = np.float32(srcPoints)
    dstPoints = np.float32(dstPoints)

    H, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    return H

def stitch(img1, img2, img3, H, G):

    h, w = img2.shape[:2]
    h2, w2 = img3.shape[:2]
    srcPoints = np.float64([[[0, 0]], [[0, h]], [[w, h]], [[w, 0]]])
    srcPoints2 = np.float64([[[0, 0]], [[0, h2]], [[w2, h2]], [[w2, 0]]])
    dstPoints = cv2.perspectiveTransform(srcPoints, H) #matches points from img2 to img1
    dstPoints2 = cv2.perspectiveTransform(srcPoints2, G) #matches points from img3 to img1

    points = np.concatenate((srcPoints,dstPoints, srcPoints2, dstPoints2), axis=0)

    x_min, y_min = np.int32(points.min(axis=0).ravel())
    x_max, y_max = np.int32(points.max(axis=0).ravel())
    
    new_w = int(np.ceil(x_max-x_min))
    new_h = int(np.ceil(y_max-y_min))

    T = np.float32([[1,0,-x_min],[0,1,-y_min],[0,0,1]])

    out1 = cv2.warpPerspective(img2,T,(new_w, new_h)) #original image but realigned to be stitched
    out2 = cv2.warpPerspective(img1,T@H,(new_w, new_h)) #left image, realigned
    out3 = cv2.warpPerspective(img3, T@G, (new_w, new_h)) #right image, realigned
    cv2.imwrite("Output\\realigned_middle.jpg", out1)
    cv2.imwrite("Output\\realigned_left.jpg", out2)
    cv2.imwrite("Output\\realigned_right.jpg", out3)
    show(out1)
    show(out2)
    show(out3)

    kernel = np.ones((30, 30), np.uint8) #using a kernel to blur the mask

    mask = cv2.warpPerspective(img2*0+255,T@H,(new_w, new_h))/255.0
    mask = cv2.erode(mask, kernel)
    mask = cv2.blur(mask,(30,30))

    mask2 = cv2.warpPerspective(img2*0+255, T@G, (new_w, new_h))/255.0
    mask2 = cv2.erode(mask2, kernel)
    mask2 = cv2.blur(mask2,(30,30))

    out = np.uint8((out1*(1-mask)+out2*mask + out1*(1-mask2)+out3*mask2)-out1)
    # out2 = np.uint8(out1*(1-mask2)+out3*mask2)
    show(out[::2, ::2])
    cv2.imwrite("Output\\pano.jpg", out)
    return out
    

img1 = cv2.imread("Input\\left01.jpg") #left
img2 = cv2.imread("Input\\middle01.jpg") #middle
img3 = cv2.imread("Input\\right01.jpg") #right

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None) 
kp2, des2 = orb.detectAndCompute(img2, None) 
kp3, des3 = orb.detectAndCompute(img3, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#good matches between L and M
goodMatches1 = goodMatches(bf.knnMatch(des1, des2, k=2), .75)
#M and R
goodMatches2 = goodMatches(bf.knnMatch(des3, des2, k=2), .75)

#save these
out1 = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches1, None)
cv2.imwrite("Output\\keypoint_matches1.jpg", out1)
out2 = cv2.drawMatches(img3, kp3, img2, kp2, goodMatches2, None)
cv2.imwrite("Output\\keypoint_matches2.jpg", out2)


H = getHomography(kp1, kp2, goodMatches1)
G = getHomography(kp3, kp2, goodMatches2)

stitch(img1, img2, img3, H, G)
