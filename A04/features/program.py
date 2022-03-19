import numpy as np
import cv2

def show(img, waitTime=0):
    # img[img > 255] = 255
    # img[img < 0] = 0
    cv2.imshow('output.png', np.uint8(img))
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

def normalize(img):
    img = img - np.min(img)
    img = img / img.max() 
    img *= 255.9999 

    return img


img = cv2.imread("features\\image2.jpg")
img = cv2.GaussianBlur(img, (11, 11), -1)
sobel = np.float64([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
Iy = cv2.filter2D(img*1.0, -1, sobel) #discrete derivative of the image in the y direction
Ix = cv2.filter2D(img*1.0, -1, sobel.T) #3x2.T = 2x3
I = (Ix**2 + Iy**2)**0.5
show(Iy)
show(Ix)
show(I) #(try show normalize(I)
show(normalize(I))

#something cool
# angle = (np.uint8((np.arctan2(Ix, Iy)+np.pi)/np.pi*90))%180
# angleView = cv2.imread("features\\image1.jpg")*0 #image with 3 empty color channels
# angleView[:, :, 0] = angle
# angleView[:, :, 1] = 255
# angleView[:, :, 2] = 255 #normalize(I)
# angleView = cv2.cvtColor(angleView, cv2.COLOR_HSV2RGB)
# show(angle)

# thing = cv2.GaussianBlur(img, (15, 15), -1)
# show(thing)

# corners = cv2.goodFeaturesToTrack(img[:, :, 1], 20, .1, 5)
# for point in corners:
#     x, y = point[0]
#     x = int(x)
#     y = int(y)
#     # cv2.circle(img, (x,y), 5, (0, 255, 0), 3)
#     cv2.drawMarker(img, (x, y), (0, 255, 0),cv2.MARKER_STAR, thickness=1)
# show(img)
