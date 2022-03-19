import numpy as np
import cv2

def show(img, waitTime=0):
    # img[img > 255] = 255
    # img[img < 0] = 0
    cv2.imshow('output.png', np.uint8(img))
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

img1 = cv2.imread("features\\image1.jpg")
img2 = cv2.imread("features\\image2.jpg") 

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None) #query image #keypoints and descriptions of those kps
kp2, des2 = orb.detectAndCompute(img2, None) #train image
# print(des1[0])
# print(des2[0])

print("1")

#finds best matches between the descriptions of the key points
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False) #brute force matcher. defs = cv2.NORM_L2, crossCheck=True
#gets two best matches.
matches = bf.knnMatch(des1, des2, k=2) #two best matches
#k-nearest neighbor

goodMatches = []
for match1, match2 in matches:
    # print(match1.distance, match2.distance) 
    if match1.distance < 0.875*match2.distance: #good matches are closer in distance
        goodMatches.append(match1)


print(len(goodMatches))
out = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, None)
# show(out) #more overlapping lines means less good matches. 
srcPoints = []
dstPoints = []
for match in goodMatches:

    index1 = match.queryIdx
    index2 = match.trainIdx

    srcPoint = kp1[index1].pt
    dstPoint = kp1[index2].pt

    srcPoints.append([srcPoint])
    dstPoints.append([dstPoints])

srcPoints = np.float32(srcPoints)
dstPoints = np.float32(dstPoints)

#this is taking forever
H, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0) #20.0 = only keep pixels that are different by 20
    #H is a transformation matrix to line H up to img1
print(H)
print("Done")

warp = cv2.warpPerspective(img1, H, None)
show(warp)

h,w = img1.shape[:2]
new_src_points = np.float64([[[0, 0]],[[0, h]], [[w, h]], [[w, 0]]])
final_points = cv2.perspectiveTransform(new_src_points, H)
#print(np.vstack(final_points, new_src_points))
points = np.vstack(final_points, new_src_points)
points = points[:, 0, :] #turns 3d thing into a 2d thing
max_x = points[:, 0].max()
max_y = points[:, 1].max()
min_x = points[:, 0].min()
min_y = points[:, 1].min()
width = int(np.ceil(max_x - min_x))
height = int(np.ceil(max_y - min_y))
T = np.float32([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])

out1 = cv2.warpPerspective(img1, T@H, (width, height)) #(cv2.add(img1, n))
out2 = cv2.warpPerspective(img2, T, (width, height))

mask = cv2.warpPerspective(img1*0+255, T@H, (width, height))/255.0
kernel = np.ones((41, 41), np.uint8) #kernel twice as big as you want to erode
mask = cv2.erode(mask, kernel)
mask = cv2.blur(mask, (41, 41))

out = np.uint8(out1*mask + out2*(1-mask)) #you want out1 where the mask is and out2 where the mask isn't 

'''
to get rid of ugly edges in the out, erode it then gaussian blur it.
'''



# matches = bf.match(des1, des2)
# # print(dir[matches[0]]) #distance, imgIdx, queryIdx, trainIdxy
# print(matches[0].distance) #smaller is better
# print(matches[0].imgIdx)
# index1 = matches[0].queryIdx
# index2 = matches[0].trainIdx
# print(des1[index1])
# print(des2[index2])

#for extra opencv camera calibration

#for making final image better:
#brightness diff map
#use it to make even brightness