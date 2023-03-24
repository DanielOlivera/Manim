import cv2 as cv
import pandas as pd
import numpy as np
import dlib
import imutils
from scipy.spatial import distance

def calculate_OAR(outline):
    A = distance.euclidean(outline[0], outline[1])
    B = distance.euclidean(outline[1], outline[2])
    C = distance.euclidean(outline[2], outline[3])
    D = distance.euclidean(outline[3], outline[4])
    E = distance.euclidean(outline[4], outline[5])
    F = distance.euclidean(outline[5], outline[6])
    G = distance.euclidean(outline[6], outline[7])
    H = distance.euclidean(outline[7], outline[8])
    I = distance.euclidean(outline[8], outline[9])
    J = distance.euclidean(outline[9], outline[10])
    K = distance.euclidean(outline[10], outline[11])
    L = distance.euclidean(outline[11], outline[12])
    M = distance.euclidean(outline[12], outline[13])
    N = distance.euclidean(outline[13], outline[14])
    O = distance.euclidean(outline[14], outline[15])
    P = distance.euclidean(outline[15], outline[16])
    Q = distance.euclidean(outline[0], outline[16])
    oar_aspect_ratio = (A+B+C+D+E+F+G+H+I+J+K+L+M+N+O+P)/(16.0*Q)
    return oar_aspect_ratio

def calculate_LBAR(brow):
    A = distance.euclidean(brow[0], brow[8])
    B = distance.euclidean(brow[1], brow[7])
    C = distance.euclidean(brow[3], brow[6])
    D = distance.euclidean(brow[4], brow[5])
    E = distance.euclidean(brow[0], brow[4])
    lbrow_aspect_ratio = (A+B+C+D)/(4.0*E)
    return lbrow_aspect_ratio

def calculate_NUAR(noseup):
    A = distance.euclidean(noseup[0], noseup[1])
    B = distance.euclidean(noseup[1], noseup[2])
    C = distance.euclidean(noseup[2], noseup[3])
    D = distance.euclidean(noseup[3], noseup[4])
    E = distance.euclidean(noseup[4], noseup[5])
    F = distance.euclidean(noseup[5], noseup[3])
    noseup_aspect_ratio = (A+B+C+D+E+F)/(6.0*F)
    return noseup_aspect_ratio

def calculate_NDAR(nosedown):
    A = distance.euclidean(nosedown[0], nosedown[1])
    B = distance.euclidean(nosedown[1], nosedown[2])
    C = distance.euclidean(nosedown[2], nosedown[3])
    D = distance.euclidean(nosedown[3], nosedown[4])
    E = distance.euclidean(nosedown[4], nosedown[5])
    F = distance.euclidean(nosedown[5], nosedown[6])
    G = distance.euclidean(nosedown[6], nosedown[7])
    H = distance.euclidean(nosedown[0], nosedown[7])
    I = distance.euclidean(nosedown[0], nosedown[3])
    nose_central = (A+B+C)
    nose_long = (nose_central+H+I)/(3.0)
    nosedown_aspect_ratio = (nose_central+D+E+F+G)/(nose_long)
    return nosedown_aspect_ratio

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (A+B)/(2.0*C)
    return eye_aspect_ratio

def calculate_OLAR(outlip):
    A = distance.euclidean(outlip[1], outlip[10])
    B = distance.euclidean(outlip[2], outlip[9])
    C = distance.euclidean(outlip[3], outlip[7])
    D = distance.euclidean(outlip[4], outlip[6])
    E = distance.euclidean(outlip[0], outlip[5])
    outlips_aspect_ratio = (A+B+C+D)/(4.0*E)
    return outlips_aspect_ratio

def calculate_ILAR(inlip):
    A = distance.euclidean(inlip[1], inlip[7])
    B = distance.euclidean(inlip[2], inlip[6])
    C = distance.euclidean(inlip[3], inlip[5])
    D = distance.euclidean(inlip[0], inlip[4])
    inlips_aspect_ratio = (A+B+C)/(3.0*D)
    return inlips_aspect_ratio

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("D:\VSCnotebook\TPfinal\landmarks.dat")

while True:
    _, frame = cap.read()
    frame = cv.flip(frame,1)
    frame = imutils.resize(frame, width=640)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        
        #outline
        Outline = []#(0,17)
        #leftEyebrow = []#(17,22)
        leftEyebrowratio = []#(17,22 & 36,40)
        #rightEyebrow = []#(22,27)
        rightEyebrowratio = []#(22,27 & 42,46)
        #Nose = []#(27,36)
        NoseratioUp =[]#(21,22,42,28,39,27)
        NoseratioDown = []#(28,29,30,31,32,33,34,35)
        #lefteye
        leftEye = []#(36,42)
        #righteye
        rightEye = []#(42,48) 
        outLips = []#(48,60)
        inLips = []#(60,68)
        #Outline
        for n in range(0, 17):
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            Outline.append((x1, y1))
            next_point = n+1
            if n == 16:
                next_point = 16
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
        #LeftEye
        for n in range(36, 42):
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            leftEye.append((x1, y1))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
        #RightEye    
        for n in range(42, 48):
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            rightEye.append((x1, y1))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
        #LeftEyebrow
        for n in range(17, 22):
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            leftEyebrowratio.append((x1, y1))
            next_point = n+1
            if n == 21:
                next_point = 39      
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
            if next_point == 39:
                for m in range(36, 40):
                   m = 75-m
                   x1 = face_landmarks.part(m).x
                   y1 = face_landmarks.part(m).y
                   leftEyebrowratio.append((x1, y1))
                   inner_next_point = m-1
                   if inner_next_point == 35:
                      inner_next_point = 17
                   x2 = face_landmarks.part(inner_next_point).x
                   y2 = face_landmarks.part(inner_next_point).y
                   cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
                   cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
        #RightEyebrow
        for n in range(22, 27):
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            rightEyebrowratio.append((x1, y1))
            next_point = n+1
            if n == 26:
                next_point = 45      
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
            if next_point == 45:
                for m in range(42, 46):
                   m = 87-m
                   x1 = face_landmarks.part(m).x
                   y1 = face_landmarks.part(m).y
                   rightEyebrowratio.append((x1, y1))
                   inner_next_point = m-1
                   if inner_next_point == 41:
                      inner_next_point = 22
                   x2 = face_landmarks.part(inner_next_point).x
                   y2 = face_landmarks.part(inner_next_point).y
                   cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
                   cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)         
        #NoseUp
        for n in range(21, 23):
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            NoseratioUp.append((x1, y1))
            next_point = n+1
            if n == 22:
                next_point = 42      
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
            if next_point == 42:
                m = 42
                o = m-14
                p=28
                q = p+11
                r=39
                s = r-18
                x1 = face_landmarks.part(m).x
                y1 = face_landmarks.part(m).y
                NoseratioUp.append((x1, y1))
                x2 = face_landmarks.part(o).x
                y2 = face_landmarks.part(o).y
                cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
                cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
                x1 = face_landmarks.part(p).x
                y1 = face_landmarks.part(p).y
                NoseratioUp.append((x1, y1))
                x2 = face_landmarks.part(q).x
                y2 = face_landmarks.part(q).y
                cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
                cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
                x1 = face_landmarks.part(r).x
                y1 = face_landmarks.part(r).y
                NoseratioUp.append((x1, y1))
                x2 = face_landmarks.part(s).x
                y2 = face_landmarks.part(s).y
                cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
                cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
                if s == 21:
                    t=27
                    x1 = face_landmarks.part(t).x
                    y1 = face_landmarks.part(t).y
                    NoseratioUp.append((x1, y1))
                    u = t+1
                    x2 = face_landmarks.part(u).x
                    y2 = face_landmarks.part(u).y
                    cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
                    cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
        #NoseDown
        for n in range(28, 31):
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            NoseratioDown.append((x1, y1))
            next_point = n+1
            if n == 30:
                next_point = 33
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
            if next_point == 33:
                for m in range(31, 36):
                    x1 = face_landmarks.part(m).x
                    y1 = face_landmarks.part(m).y
                    NoseratioDown.append((x1, y1))
                    bridge_next_point = m+1
                    if bridge_next_point == 36:
                        bridge_next_point = 28
                    x2 = face_landmarks.part(bridge_next_point).x
                    y2 = face_landmarks.part(bridge_next_point).y
                    cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
                    cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
                    if bridge_next_point == 28:
                        p1 = 28
                        x1 = face_landmarks.part(p1).x
                        y1 = face_landmarks.part(p1).y
                        p2 = p1+3 
                        x2 = face_landmarks.part(p2).x
                        y2 = face_landmarks.part(p2).y
                        cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
        #outLips(48,60)
        for n in range(48, 60):
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            outLips.append((x1, y1))
            next_point = n+1
            if n == 59:
                next_point = 48
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
        #inLips(60,68)
        for n in range(60, 68):
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            inLips.append((x1, y1))
            next_point = n+1
            if n == 67:
                next_point = 60
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv.circle(frame, (x1, y1), 3, (173,255,47), 1)
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
        #FUN
        outline_OAR = calculate_OAR(Outline)
        outline_OAR = round(outline_OAR,4)  
        left_LBAR = calculate_LBAR(leftEyebrowratio)
        left_LBAR = round(left_LBAR,4)
        right_RBAR = calculate_LBAR(rightEyebrowratio)
        right_RBAR = round(right_RBAR,4)
        noseup_NUAR = calculate_NUAR(NoseratioUp)
        noseup_NUAR = round(noseup_NUAR,4)
        nosedown_NDAR = calculate_NDAR(NoseratioDown)
        nosedown_NDAR = round(nosedown_NDAR,4)
        left_EAR = calculate_EAR(leftEye)
        left_EAR = round(left_EAR,4)
        right_EAR = calculate_EAR(rightEye)
        right_EAR = round(right_EAR,4)
        outlips_OLAR = calculate_OLAR(outLips)
        outlips_OLAR = round(outlips_OLAR,4)
        inlips_ILAR = calculate_ILAR(inLips)
        inlips_ILAR = round(inlips_ILAR,4)
        outline_OAR_array = np.asarray(outline_OAR)
        left_LBAR_array = np.asarray(left_LBAR)
        right_RBAR_array = np.asarray(right_RBAR)
        noseup_NUAR_array = np.asarray(noseup_NUAR)
        nosedown_NDAR_array = np.asanyarray(nosedown_NDAR)
        left_EAR_array = np.asarray(left_EAR)
        right_EAR_array = np.asarray(right_EAR)
        outlips_OLAR_array = np.asarray(outline_OAR)
        inlips_ILAR_array = np.asarray(inlips_ILAR)
        sample_array = np.vstack((outline_OAR_array, left_LBAR_array, right_RBAR_array, noseup_NUAR_array,nosedown_NDAR_array, left_EAR_array, right_EAR_array, outlips_OLAR_array, inlips_ILAR_array))
        sample_array = sample_array.transpose()
        #pd.DataFrame(sample_array).to_csv('D:\VSCnotebook\TPfinal\\neutral.csv', mode='a', index=False, header=False)
        #pd.DataFrame(sample_array).to_csv('D:\VSCnotebook\TPfinal\\sorpresa.csv', mode='a', index=False, header=False)
        #pd.DataFrame(sample_array).to_csv('D:\VSCnotebook\TPfinal\\enojo.csv', mode='a', index=False, header=False)
        #pd.DataFrame(sample_array).to_csv('D:\VSCnotebook\TPfinal\\guest.csv', mode='a', index=False, header=False)
        print(sample_array)
          
    cv.imshow("", frame)

    key = cv.waitKey(1)
    if key == 27:
        break
cap.release()
cv.destroyAllWindows()