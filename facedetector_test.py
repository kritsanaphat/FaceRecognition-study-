import cv2
import keyboard
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('C:\\Users\\Kritsanaphat\\OneDrive - KMITL\\Desktop\\faceFile\\haarcascade_frontalface_default.xml')
while True:
    if keyboard.is_pressed('q'):
        break
    else:
        _, frame = cap.read()
        gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,1.1,4)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imshow('frame',frame)
            cv2.waitKey(1)
