import numpy as np
import cv2

def show(img, waitTime=0,forceNormalization=False):
    if forceNormalization or img.min()<0 or img.max()>255:
        img=normalize(img)
    cv2.imshow('image', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def normalize(img):
    img=img-np.min(img)
    img/=img.max()
    img*=255.99999
    return np.uint8(img)

# img1=cv2.imread("Pano\\image1.jpg")
# img2=cv2.imread("Pano\\image2.jpg")
img2=cv2.imread("Input\\right01.jpg")
img1=cv2.imread("Input\\middle01.jpg")

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# print(len(des1))
# print(len(des2))

bf = cv2.BFMatcher(cv2.NORM_L2) #cv2.NORM_HAMMING , crossCheck=True

matches = bf.knnMatch(des1,des2, k=2)
#k-nearest neighbor
goodMatches=[]
for match1,match2 in matches:
    if match1.distance<.75*match2.distance:
        goodMatches.append(match1)
# print(len(goodMatches))
out=cv2.drawMatches(img1, kp1, img2, kp2, goodMatches,None)
# cv2.imwrite("out.png",out)
# show(out)
srcPoints=[]
dstPoints=[]
for match in goodMatches:
    index1=match.queryIdx
    index2=match.trainIdx
    srcPoint=kp1[index1].pt
    dstPoint=kp2[index2].pt
    srcPoints.append([srcPoint])
    dstPoints.append([dstPoint])
srcPoints=np.float32(srcPoints)
dstPoints=np.float32(dstPoints)
# ~ srcPoints=cv2.cornerSubPix(	img1[:,:,1], srcPoints,winSize=(5,5),zeroZone=(-1,-1),criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
# ~ dstPoints=cv2.cornerSubPix(	img2[:,:,1], dstPoints,winSize=(5,5),zeroZone=(-1,-1),criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
# ~ print(srcPoints)
# ~ srcPoints = np.float32([kp1[m.queryIdx] 
                # ~ .pt for m in goodMatches]).reshape(-1, 1, 2) 
 
# ~ dstPoints = np.float32([kp2[m.trainIdx] 
                # ~ .pt for m in goodMatches]).reshape(-1, 1, 2) 
# ~ print(srcPoints)
H,_=cv2.findHomography(srcPoints,dstPoints, cv2.RANSAC, 5.0)
# print("Done")
# input()
out2=cv2.warpPerspective(img1,H,None)
show(out2[::2, ::2])
# cv2.imwrite("warp.png",out2)
# # ~ print(matches)#'distance', 'imgIdx', 'queryIdx', 'trainIdx'

h,w=img1.shape[:2]
srcPoints=np.float64([[[0,0]],[[0,h]],[[w,h]],[[w,0]]])
dstPoints=cv2.perspectiveTransform(srcPoints,H)
points=np.vstack((dstPoints,srcPoints))
# print(points)
points=points[:,0,:]
# print(points)
# print(points[:, 0])
max_x=points[:,0].max()
max_y=points[:,1].max()
min_x=points[:,0].min()
min_y=points[:,1].min()
# print(min_x,min_y,max_x,max_y)
width=int(np.ceil(max_x-min_x))
height=int(np.ceil(max_y-min_y))
# print(width,height)
T=np.float32([[1,0,-min_x],[0,1,-min_y],[0,0,1]])
# img1+=12
# img1[img1<12]=255
out1=cv2.warpPerspective(img1,T@H,(width,height))
# show(out1)
mask=cv2.warpPerspective(img1*0+255,T@H,(width,height))/255.0
kernel = np.ones((41, 41), np.uint8)
  
# # Using cv2.erode() method 
mask = cv2.erode(mask, kernel)
mask=cv2.blur(mask,(41,41))



out2=cv2.warpPerspective(img2,T,(width,height))
# # cv2.imwrite("big1.png",out1)
# # cv2.imwrite("big2.png",out2)
# diff=np.uint8(np.absolute(out1*1.0-out2))
# cv2.imwrite("diff.png",diff)
out=np.uint8(out1*mask+out2*(1-mask))
show(out[::2, ::2])
# # cv2.imwrite("big.png",out)

# #oriented feature from accelerated segment test and rotated binary roboust independent elem. feat.
