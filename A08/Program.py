import cv2
import numpy as np
import os


def loadHats():
    try:
        entries = os.scandir("hats/")
        hats = []
        for entry in entries:
            hats.append(cv2.imread(f"hats\\{entry.name}"))
        
        print(len(hats))
        return hats
    except:
        return cv2.imread("hats\\hat-1.png")

def drawHat(img, face_w, face_h, x, y, hat):
    h, w = img.shape[:2]
    hat_h, hat_w = hat.shape[:2]
    x -= face_w//3
    y -= face_h//2

        
    hat = cv2.resize(hat, (int(face_w*0.8), int(face_h*0.8)), interpolation=cv2.INTER_AREA)
    hat_h, hat_w = hat.shape[:2]
    view = img[y:y+hat_h, x:x+hat_w]
    try:
        view[hat!=255] = 0
    except:
        return

def on_change(n):
    N[0] = n

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap_width = np.int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #640
cap_height = np.int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #480

hats = loadHats()
cv2.namedWindow("cam")
cv2.createTrackbar("Hat Selection", "cam", 0, len(hats)-1, on_change)
N = [0]
# print(hats)

while True:
    ret, frame = cap.read()
    f_h, f_w = frame.shape[:2]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grey,scaleFactor=1.1)

    for (x, y, w, h) in faces:
        # if y > f_h*.9 or x > f_w*.9 or x < f_w*.1 or y < f_h*.1:
        #     continue
        drawHat(frame, w, h, x+w//2, y-20, hats[N[0]])

    cv2.imshow('cam', frame)

    c = cv2.waitKey(1)

    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()