from typing import Text
import face_recognition as face
from face_recognition.api import face_distance, face_encodings, face_locations
import numpy as np
import cv2

video_cp = cv2.VideoCapture("sanchoDebut.mp4")
#video_cp = cv2.VideoCapture("C:\\Users\\Kritsanaphat\\Documents\\coding\\Python Code\\FaceRecognition\\3dout.mp4")
#video_cp = cv2.VideoCapture(0)

sancho_img1 = face.load_image_file("sancho_1.jpg")
sancho_face_encoding1 = face.face_encodings(sancho_img1)[0]
'''sancho_img2 = face.load_image_file("C:\\Users\\Kritsanaphat\\Documents\\coding\\Python Code\\FaceRecognition\\sancho_2.jpg")
sancho_face_encoding2 = face.face_encodings(sancho_img2)[0]
'''

face_locations = []
face_encodings = []
face_name = []
face_percent = []
process_this_frame = True

known_face_encodings1 = [sancho_face_encoding1]
#known_face_encodings2 = [sancho_face_encoding2]
known_face_name = ["Sancho25"]

while True:
    ret , frame = video_cp.read()
    if ret:
        small_frame = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        rgb_small_frame = small_frame[:,:,::-1]
        face_name=[]    #clear this frame
        face_percent=[] #clear this frame
        if process_this_frame:
            face_locations= face.face_locations(rgb_small_frame)
            face_encodings = face.face_encodings(rgb_small_frame,face_locations)

            for face_encoding in face_encodings:
                face_distance1 = face.face_distance(known_face_encodings1,face_encoding) #Check same face
                #face_distance2 = face.face_distance(known_face_encodings2,face_encoding) #Check same face

                best = np.argmin(face_distance1)
                face_percent_value = 1-(face_distance1[best])

                if face_percent_value>=0.5:
                    name = known_face_name[best]
                    percent = round(face_percent_value*100,2)
                    face_percent.append(percent)
                    print("Yes")
                    print(face_distance1[best])
                    setName = True
                else:
                    setName = False
                    name ="Unknow"
                    face_percent.append(0)
                    print("No")
                face_name.append(name)

        for(top,right,bottom,left),naem,percent in zip(face_locations,face_name,face_percent):
            top*=2
            right*=2
            bottom*=2
            left*=2


            if setName == False:
                color = [51,51,255] #red in rgb syntax
            else:
                print("Blue")
                color = [255,51,51]


            cv2.rectangle(frame,(left,top),(right,bottom),color,2)
            cv2.rectangle(frame,(left-1,top -30),(right+1,top),color,cv2.FILLED)
            cv2.rectangle(frame,(left-1,bottom),(right+1,bottom+30),color,cv2.FILLED)
            
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame,name,(left+6,top-6),font,0.8,(255,255,255),1)
            cv2.putText(frame,"MATCH: "+str(percent)+"%",(left+6,bottom+23),font,0.8,(255,255,255),1)


        cv2.imshow("Video",frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'): #parameter of waitKey affect to speed in video
            break
    else:
        print("Error from Upload Video!!")
        break
video_cp.release()
cv2.destroyAllWindows()
    