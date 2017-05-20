import cv2  #import openCV library

cam = cv2.VideoCapture(0) #capture from default camera

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#importing trained dataset for detecting face

Id=raw_input('enter your id: ')
#Enter any number for naming images

sampleNum=0
#initialise the sample value

while(True):                                   #while camera is capturing
    ret, img = cam.read()                      #Read from Camera
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #captured image to Gray scale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #face detector to detect faces in our captured image
    for (x,y,w,h) in faces:                          #x,y,w,h will be defined from above code
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)    #the ractangle will be created arround the detected face
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w]) #

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
