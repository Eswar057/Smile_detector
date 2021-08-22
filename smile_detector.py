import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates= face_detector.detectMultiScale(frame_grayscale)
    
    for(x,y,w,h) in face_coordinates:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(100, 200, 50), 4)

        the_face = frame[y:y+h , x:x+w]
        
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_coordinates= smile_detector.detectMultiScale(face_grayscale, 1.7, 20)

        #eye_coordinates= eye_detector.detectMultiScale(face_grayscale, 1.1, 10)

        #for (x_, y_, w_, h_) in eye_coordinates:
            #draw all the rectangles around the eye
            #cv2.rectangle(the_face,(x_,y_),(x_+w_,y_+h_),(255, 255, 255), 4)
        
        if len(smile_coordinates) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
            fontFace = cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
    
    cv2.imshow('Smile Detector', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113 or key == 13 or key == 9:
            break


#CleanUp
webcam.release()
cv2.destroyAllWindows()

print("Code Completed")