import cv2

faceCascade = cv2.CascadeClassifier("Seward Code\\haarcascade_frontalface_default.xml")


cap = cv2.VideoCapture(0)
while True:
    frame = cap.read()[1]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grey,scaleFactor=1.1)
    print(len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 5)
    # ~ # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
