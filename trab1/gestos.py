import numpy as np
import cv2 as cv
import pyautogui
import time
pyautogui.FAILSAFE = False
paramsShiTomasi = dict( maxCorners = 40,
                       qualityLevel = 0.9,
                       minDistance = 1,
                       blockSize = 10)
paramsLukasKanade = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

def setas(dir):
    pyautogui.press(dir)
    print(dir)
    time.sleep(2)
def iniciar(cap):
    global frame_cinzaIn, p0, mask
    ret, frameIn = cap.read()
    frame_cinzaIn = cv.cvtColor(frameIn, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(frame_cinzaIn, mask = None, **paramsShiTomasi)
    mask = np.zeros_like(frameIn)
color = np.random.randint(0, 255, (100, 3))
qntdAguardoFrames = 100
qntdAtualFrames = 0
def loop_gestos(cap, ret, frame):
    global frame_cinzaIn, p0, mask, qntdAtualFrames
    frame_cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(frame_cinzaIn, frame_cinza, p0, None, **paramsLukasKanade)
    qntdEsq = 0
    qntdEsq = 0
    qntdIgual = 10
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        qntdIgual = 0
        qntdEsq= 0
        qntdDir = 0
        for i in range(0, len(good_new)):
            
            if(abs(good_new[i][0] - good_old[i][0]) >= 0.1):
                if(good_new[i][0]-good_old[i][0] > 0):
                    qntdEsq += 1
                else:
                    qntdDir += 1
            else:
                qntdIgual += 1
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        frame_cinzaIn = frame_cinza.copy()
        p0 = good_new.reshape(-1, 1, 2)
    setas("left") if qntdEsq > qntdDir else setas('right') if qntdDir >= qntdIgual else print('Parado')
    img = cv.add(frame, mask)
    
    # return frame
    