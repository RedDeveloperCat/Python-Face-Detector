import cv2
from random import randrange

# loading pre-trained data from opencv (haarcascade)
# classifier is just detector

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# To capture video from existing video.   
cap = cv2.VideoCapture('ST4.mp4')  
  
while True:  
    # Read the frame  
    _, img = cap.read()  
  
    # Convert to grayscale  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
    # Detect the faces  
    faces = trained_face_data.detectMultiScale(gray, 1.1, 4)  
  
    for (x, y, w, h) in faces:  # loop to show all faces
        # create rectangles around face and randrage here creates random colors for rectangles
        cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(128, 256),randrange(128, 256), randrange(128, 256)), 10)
  
    # Display  
    cv2.imshow('Face Detector but in Video', img)  
  
    # Stop if escape key is pressed  
    k = cv2.waitKey(30) & 0xff  
    if k==27:  
        break  
          
# Release the VideoCapture object  
cap.release()  

print('Trippin through times lol... but code finished')
