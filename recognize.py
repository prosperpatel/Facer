import cv2
import numpy as np

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')   #load trained sample
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");   #load haarcascde


cam = cv2.VideoCapture(0)   #capture from default camera
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1                   #width
fontColor = (255, 255, 255) 
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
    
        if(Id==1):
                Id="Prosper"
           
        else:
            Id="Unknown"
        
        cv2.putText(im,str(Id),(x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        
        
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
