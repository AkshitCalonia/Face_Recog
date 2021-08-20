import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier('sourcz.xml')

# getting the web cam vid.
webcam = cv2.VideoCapture(0)
# 0 means the default webcam live video 

while True:

    succesful_frame_read, frame = webcam.read()
# to get the image 
# img = cv2.imread('RDJ.jpg')
# img = cv2.imread('multiti.jpg')


    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face 
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


    # draw rectangle s arounf theh fac
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256), randrange(256), randrange(256)), 3)


    cv2.imshow('face detect', frame)

    key = cv2.waitKey(1)

    # using the ascii key for caps and small "q"
    if key==81 or key==113 or key==27:
        break


webcam.release()
### stoping the webcam
