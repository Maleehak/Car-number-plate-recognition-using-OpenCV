import cv2
import numpy as np

carPlatesCascade = cv2.CascadeClassifier('haarcascades/haarcascade_russian_plate_number.xml')

cap = cv2.VideoCapture('carVideo.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 80)


if (cap.isOpened()==False):
    print('Error Reading video')

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    car_plates = carPlatesCascade.detectMultiScale(gray,scaleFactor=1.2,
    minNeighbors = 5, minSize=(25,25))

    for (x,y,w,h) in car_plates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        plate = frame[y: y+h, x:x+w]
        plate = cv2.blur(plate,ksize=(20,20))
        # put the blurred plate into the original image
        frame[y: y+h, x:x+w,2] = plate

    if ret == True:
        cv2.imshow('Video',frame)
    
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    else:
        break

cap.release()
cv2.destroyAllWindows()
