import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

Y, X = np.float32(np.mgrid[:360, :640])
i=0
last = time.time()
fps = 30
YP, XP = Y, X
n = 4
while True:
    i+=1
    ret, frame = cap.read()

    #wavy
    # YP = Y+10*np.sin((X+3+i)/100*2*np.pi)
    # theta = X/640*2*np.pi
    # r = Y/2
    # XP = r*np.cos(theta)+320 #cartestian to polar
    # YP = r*np.sin(theta)+240
    
    #circle
    # X1 = X-320 #center x
    # Y1 = Y-240 #center y
    # r = np.hypot(X1, Y1) #distance from the center
    # theta = np.arctan2(Y1, X1) #list of angles for different distances
    # XP = theta/(2*np.pi)*640%640
    # YP = r*2

    #water ripple 
    # X1 = X-320
    # Y1 = Y-240
    # r = np.hypot(X1, Y1)
    # theta = np.arctan2(Y1, X1)
    # dr = (-r/40+30)*np.sin(r/50*2*np.pi-i/10.0)
    # r += dr
    # XP = r*np.cos(theta)+320
    # YP = r*np.sin(theta)+240

    XP = (X*n+200*np.sin(time.time()/2.5*2*np.pi))%640
    YP = Y*n%360

    frame = cv2.remap(frame, XP, YP, cv2.INTER_LINEAR)

    current = time.time()
    dt = current - last
    last = current 
    # fps = fps *.99(1/dt)*.01 #easy way to smooth data. 1/dt is current fps and fps is expected. 
    print(fps)

    # cv2.imshow("Input", np.uint8(frame-dr[:, :, None])) #for ripple
    cv2.imshow("Input", frame) #subtract a number to darken

    c = cv2.waitKey(1)
    print(c)
    if c == 97:
        c += 1
    if c == 100:
        c -= 1
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()