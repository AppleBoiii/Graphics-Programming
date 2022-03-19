import numpy as np
import cv2
def show(img, waitTime=0):
    cv2.imshow('output.png', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

img = cv2.imread('image1.png')
h, w = img.shape[:2]
Y, X = np.mgrid[:h, :w]
C = X*0+1
# X = np.dstack((X,Y,C))
points = np.stack((X.ravel(),Y.ravel(),C.ravel())) #creates a matrix that looks like T
print(points)

angle = -np.pi/6
s = np.sin(angle)
c = np.cos(angle)

#translates image to T1, rotates it, then translates it back with T2.
# T1 = np.float64([[1, 0, -w/2], \
#                 [0, 1, -h/2], \
#                 [0, 0, 1]])

# T2 = np.float64([[1, 0, w/2], \
#                 [0, 1, h/2], \
#                 [0, 0, 1]])

# R = np.float64([[c, -s, 0], \
#                 [s, c, 0], \
#                 [0, 0, 1]]) #T = transformation matrix, x coordinate += 10

# Mirror = np.float64([[1, 0, 0], \
#                     [0, -1, h-1], \
#                     [0, 0, 1]])

# M = T2@R@T1 #reads from right to left (not communative)

# out = cv2.warpPerspective(img, Mirror, None)
# show(out)


# x = np.float64([5, 10, 1])
# Xp = np.int32(T@points) #dot product T and points and don't do uint8 bc we want signed integers 
# out = img*0
# Y, X, _ = Xp
# goodPixel = (Y<0)*(Y>=h)*(X<0)*(X>=w)

# out[points[1]][goodPixel], points[0][goodPixel] = img[Y[goodPixel], X[goodPixel]]
# # out[Y, X] = img[points[1], points[0]]
# show(out)


#making trapezoid from image (perspective transformation)

#while True:
    # angle = 0 
    # s = np.sin(angle*np.pi/180)
    # c = np.cos(angle*np.pi/180)
    # R = np.float64([[c, -s, 0], \
    #                 [s, c, 0], \
    #                 [0, 0, 1]]) 
    # out = cv2.warpPerspective(img, M@R, None)
    # show(out, 33)
    # angle += 10

# srcPoint = np.float32() #original coords
# dstPoint = np.float32() #where you want them to be after
# M = cv2.getPerspectiveTransform()

'''
out = something
mask = cv2.warpPerspective(img*0+255, M, None)
show(np.uint8(img*(1-mask)+out*mask)
'''
#get perspective transform
# srcPoint = np.float32() #original coords
# dstPoint = np.float32() #where you want them to be after
# M = cv2.getPerspectiveTransform()
# out = cv2.warpPerspective, img1, M, None, flags=cv2.INTER_Nearest)
#can't just add images
#mask = out > 0 (this is a bad choice for a mask when by itself)
#instead make a version of the mask (the transformed image) that is the image but only white -> mask = cv2.warpPerspective(img*0+255, M, None, flags=cv2.INTER_NEAREST)
# mask = mask > 0
# show(np.uint8(out*mask+img1*(1-mask))) #has a few problems with it

#another way to fix the anti-aliasing and make the composite better
#whiteOutVersion = img1*0+255
#whiteOutVersion[:10] = 0
#whiteOutVersion[-10:0] = 0
#whiteOutVersion[:, :10] = 0
#whiteOutVersion[:, -10:0] = 0
#mask = warPerspective(whiteOutVersion)
#show(np.uint8(out*mask/255.0+img1*(255-mask)/255.0)) #has a few problems with it