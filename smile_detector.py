
import cv2

#load some pre-trained data on face frontals from opencv
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')


#capture the video
webcam = cv2.VideoCapture(0) #0- mean the default camera

#Iterate forever over frames
while True:
    successful_frame_read, frame = webcam.read()

    #if there is an error about
    if not successful_frame_read:
        break


#convert img to greyscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


#detect faces
    faces = face_detector.detectMultiScale(frame_grayscale)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

        the_face = frame[y:y+h ,x:x+w]

        # face_grayScale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        face_grayScale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face_grayScale, scaleFactor=1.7, minNeighbors=20)

        eyes = eye_detector.detectMultiScale(face_grayScale, scaleFactor=1.1, minNeighbors=10)

        #find all smiles
        for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_, h_), (50, 50, 200), 4)
            
            
        for (x_, y_, w_, h_) in eyes:
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_, h_), (255, 255, 255), 4)


        if len(smiles) > 0 :
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255 ,255))
                        

    cv2.imshow('Smile Detector', frame)


    key= cv2.waitKey(1) #wait 1 milisceond then go to the next iteration

    #Stop if Q key is pressed
    if key==81 or key== 113:
        break

    #Release the video capture object
webcam.release()
cv2.destroyAllWindows()


print("code completed!!!")