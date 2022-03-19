import cv2
import numpy as np
import time 

def normalize(img):
    img = img*1.0
    img = img-np.min(img)
    img /= img.max()
    img *= 255.99999
    return np.uint8(img)

def gaussianNoise(img, loc=0, scale=10):
    gauss = np.random.normal(loc, scale, img.shape)
    gauss = gauss.reshape(img.shape)
    # print(gauss)
    noise = gauss + img

    return noise

def tint(img, color=(0, 255, 0), percent=0.5):
    tint = img*1
    tint[:, :] = color
    tinted_img = (tint*percent)+img*(1-percent)

    return tinted_img

def glow(img, strength=1, radius=10):
    if not radius%2: #since radius is used in blur it has to be an odd kernel
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 1)
    blended = cv2.addWeighted(img, 1, blur, strength, 0) #addWeighted alpha blends the blurred image onto the original

    return blended

def nightVision(img):
    #nightvision
    #add grain / noise
    #tint green
    #add glow
    img = glow(img, 1.2, 21)
    img = gaussianNoise(img, 0, 25)
    img = tint(img, (0, 255, 0), 0.5)
    img = glow(img, 1.5, 30)
    img = normalize(img)

    return img

def wonk(img, X, Y):
    YP, XP = Y*1, X*1
    h, w = img.shape[:2]
    
    # YP = (np.sin((X*np.pi)/100))*Y/2 #this kinda cool
    temp = (np.sin((X[h//4:h//4+h//2])/(h/2)))*h/3.5
    YP[h//4:h//4+h//2] = temp

    # center_y, center_x = h//2, w//2
    # r = np.hypot(center_y, center_x)
    # thetas = np.arctan2(center_y, center_x)

    # # theta = X/w*2*np.pi
    # # r = min(h, w)
    # XP = r*np.cos(thetas)
    # YP = r*np.sin(thetas)

    frame = cv2.remap(img, XP, YP, cv2.INTER_LINEAR)
    return frame

def perspectiveWarp(frame):
    h, w = frame.shape[:2]
    srcPoints = np.float32([(0, 0), (0, h), (w, h), (w, 0)])
    dstPoints = np.float32([((w//2)-20, h//2), (0, h), (w, h), ((w//2)+20, h//2) ])
    T = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    n_frame = cv2.warpPerspective(frame, T, (w, h), flags=cv2.INTER_NEAREST)
    n_frame[:h//2, :] = n_frame[h//2:, :]

    return n_frame

def gridify(frame, X, Y, nx=2, ny=2):
    XP, YP = X*1, Y*1
    h, w = frame.shape[:2]

    YP = Y*ny%h
    XP = X*nx%w

    frame = cv2.remap(frame, XP, YP, cv2.INTER_LINEAR)
    return frame

def on_change_1(n):
    if not n:
        n = 1
    N[0] = n

def on_change_2(n):
    if not n:
        n = 1
    N[1] = n


def mainloop(cap):
    global N
    cap_width = np.int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #640
    cap_height = np.int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #480

    Y, X = np.float32(np.mgrid[:cap_height, :cap_width])
    YP, XP = Y, X

    cv2.namedWindow("cam")
    # cv2.createTrackbar('X-Squares', "cam", 1, 30, on_change_1)
    # cv2.createTrackbar('Y-Squares', "cam", 1, 30, on_change_2)
    N = [1, 1]
    FLAG_1 = False
    FLAG_2 = False
    FLAG_3 = False
    FLAG_4 = False


    last = time.time()
    fps = 30
    while True:
        ret, frame = cap.read()
        if FLAG_1:
            frame = nightVision(frame)
        if FLAG_2:
            frame = wonk(frame, X, Y )
        if FLAG_3:
            frame = perspectiveWarp(frame)
        if FLAG_4: 
            frame = gridify(frame, X, Y, N[0], N[1])
        
        current = time.time()
        dt = current - last
        last = current 
        fps = fps*.99+(1/(dt+.01))*.01  #easy way to smooth data. 1/dt is current fps and fps is expected. 
        text = str(int(fps))
        cv2.putText(frame, text, org=(20, cap_height-50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        cv2.imshow("cam", frame)
        c = cv2.waitKey(1)
        # print(c)

        if c == 49:
            FLAG_1 = not FLAG_1
        if c == 50:
            FLAG_2 = not FLAG_2
        if c == 51:
            FLAG_3 = not FLAG_3
        if c == 52:
            FLAG_4 = not FLAG_4
            if FLAG_4:
                cv2.createTrackbar('X-Grid', "cam", 1, 30, on_change_1)
                cv2.createTrackbar('Y-Grid', "cam", 1, 30, on_change_2)
        if c == 27:
            break

cap = cv2.VideoCapture(0)
print("Press keys to toggle on/off stuff. Filters can be combined. \n1 - Fake NVG \n2 - Wonk Filter \n3 - Warpy Filter \n4 - Grid Grid Grid Grid Grids \nEsc - Leave")
mainloop(cap)
cap.release()
cv2.destroyAllWindows()