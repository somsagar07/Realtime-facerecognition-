import cv2
import face_recognition
import tkinter as tk
import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
global frp,tname#list of paths
frp=[]
tname=[]

def getfilename(path):
    a=path.split(r'/')
    fname=a[-1]
    a=fname.split('.')
    a=a[0]
    return a

def openfilename():
    filename = filedialog.askopenfilename(title ='"pen') 
    return filename 


def train1():
    path = openfilename() 
    name=getfilename(path)
    if len(tname)!=2:
        tname.append(name)
        frp.append(path)
    else:
        tname[0]=name
        frp[0]=path
    print(tname,frp)

def train2():
    path= openfilename() 
    name=getfilename(path)
    if len(tname)!=2:
        tname.append(name)
        frp.append(path)
    else:
        tname[1]=name
        frp[1]=path
    print(tname,frp)
    
def train3():
    path = openfilename() 
    name=getfilename(path)
    if len(tname)!=2:
        tname.append(name)
        frp.append(path)
    else:
        tname[2]=name
        frp[2]=path
    print(tname,frp)
    

###########################
def face_reco(): 
    video_capture = cv2.VideoCapture(0)
    if len(tname)==0:
        train1()
    # Load a sample picture and learn how to recognize it.
    if len(tname)==1:
        image1 = face_recognition.load_image_file(frp[0])
        image1_face_encoding = face_recognition.face_encodings(image1)[0]
        known_face_encodings = [
                image1_face_encoding
        ]
        known_face_names = [
                tname[0]
        ]
    elif len(tname)==2:
        image1 = face_recognition.load_image_file(frp[0])
        image1_face_encoding = face_recognition.face_encodings(image1)[0]
        image2 = face_recognition.load_image_file(frp[1])
        image2_face_encoding = face_recognition.face_encodings(image2)[0]
        known_face_encodings = [
                image1_face_encoding,
                image2_face_encoding
        ]
        known_face_names = [
                tname[0],
                tname[1]
        ]
    elif len(tname)==3:
        image1 = face_recognition.load_image_file(frp[0])
        image1_face_encoding = face_recognition.face_encodings(image1)[0]
        image2 = face_recognition.load_image_file(frp[1])
        image2_face_encoding = face_recognition.face_encodings(image2)[0]
        image3 = face_recognition.load_image_file(frp[2])
        image3_face_encoding = face_recognition.face_encodings(image3)[0]
        known_face_encodings = [
                image1_face_encoding,
                image2_face_encoding,
                image3_face_encoding
        ]
        known_face_names = [
                tname[0],
                tname[1],
                tname[2]
        ]
    
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
        
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
        
            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
        
                    face_names.append(name)
        
            process_this_frame = not process_this_frame
        
        
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
            # Display the resulting image
            cv2.imshow('Video2', frame)
        
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # Release handle to the webcam
    video_capture.release() 
###########################
    
win = tk.Tk()
win.resizable(width = True, height = True) 
menubar = Menu(win)

           
# Adding File Menu and commands
file = Menu(menubar, tearoff=0)
face_rec=Menu(menubar,tearoff=0)

menubar.add_cascade(label='Face Recognition',menu=face_rec, command=face_reco)
face_rec.add_command(label='Execute', command=face_reco)
face_rec.add_command(label="Train 1", command=train1)
face_rec.add_command(label="Train 2", command=train2)
face_rec.add_command(label="Train 3", command=train3)

#======================
# Start GUI
#======================

win.config(menu = menubar)
win.mainloop()