import cv2

CASCADE_PATH = ""
faceCascade = cv2.CascadeClassifier_convert(CASCADE_PATH) #get face cascade xml classifier on opencv github

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        # minNeighbors = 3,
        # minSize = (30, 30),
        # flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()