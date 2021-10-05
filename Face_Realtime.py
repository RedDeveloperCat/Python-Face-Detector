import cv2
from random import randrange

# loading pre-trained data from opencv (haarcascade)
# classifier is just detector

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose image to detect face

# img = cv2.imread('mrrobot.jpg') #this is how you import image in opencv

webcam = cv2.VideoCapture(0)  # capturing live video

# loop to capture video
while True:

    successful_frame_read, frame = webcam.read()

    # we need to convert to grayscale before detecting faces

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # we will detect faces using the line below
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:  # loop to show all faces
        # create rectangles around face and randrage here creates random colors for rectangles
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256),randrange(128, 256), randrange(128, 256)), 10)
                  
    # this is app name for window and taking the img
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()

# #we will detect faces using the line below
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# # here im using coordinates to accurately crop rectangle around face
# # (x,y,w,h) = face_coordinates[0] #change this [0] no to 1,2,3 to switch to other faces on images

# for (x, y, w, h) in face_coordinates: # loop to show all faces
#     cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(128,256), randrange(128,256),randrange(128,256)), 10) #create rectangles around face and randrage here creates random colors for rectangles

# # print(face_coordinates)


# cv2.imshow('Face Detector', img)#this is app name for window and taking the img
# cv2.waitKey()


print('Trippin through times lol... but code finished')
