import cv2
import numpy as np
import math

def normalize(img):
    img = img*1.0
    img = img-np.min(img)
    img /= img.max()
    img *= 255.99999
    return np.uint8(img)

def show(img, waitTime=0):
    cv2.imshow('output.png', img)
    cv2.waitKey(waitTime)
    if not waitTime:
        cv2.destroyAllWindows()

img = 255-cv2.imread("img3.png",0)
# _, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY, -1)
show(img)

contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(hierarchy)
#[nextSib, prevSib, firstChild, parent]

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
UNIQUE_OBJECTS = []
for i, c in enumerate(contours):
    cv2.drawContours(output, contours, i, (0, 255, 0), 2)
    m = cv2.moments(c)
    hu = cv2.HuMoments(m)
    for j, moment in enumerate(hu):
        hu[j] = math.log(abs(moment), 10)*math.copysign(1.0, moment)*-1
    # (lambda x: [print(abs(int(x[j]))) for j in range(len(x))])(hu)
    # print(f"Shape{i}")
    # print("---")

    x = m['m10']/m['m00']
    y = m['m01']/m['m00']
    r = (m['m00']**.5)/2

    h = abs(int(hu[-1]))
    print(hu)
    if h not in UNIQUE_OBJECTS:
        UNIQUE_OBJECTS.append(h)
    
    cv2.circle(output, (int(x), int(y)), 3, (0, 255, 0), 3)
    cv2.putText(output, str(UNIQUE_OBJECTS.index(h)), (int(x+r), int(y+r)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, 2)

    
show(output)
print(UNIQUE_OBJECTS)



# params = cv2.SimpleBlobDetector_Params()
# params.minThreshold = 1
# params.maxThreshold = 255
# params.filterByArea = True
# params.minArea = 1

# detector = cv2.SimpleBlobDetector_create()
# keypoints = detector.detect(img)
