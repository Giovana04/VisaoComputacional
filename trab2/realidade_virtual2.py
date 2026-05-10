import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import cubo


model_path = 'trab2\\hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("A câmera não pode ser aberta!")
    exit()
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("A câmera não pode ser aberta!")
            break
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        # Detecta mão
        # Detecta apenas uma mão!! Se mais de uma mão estiver na imagem, apenas reconhece a primeira que apareceu.
        hand_landmarker_result = landmarker.detect(mp_image)
        esq = None
        dir = None
        baixo = None
        cima = None
        
        for hand_mark in hand_landmarker_result.hand_landmarks:
            for i in range(0, len(hand_mark)):
                pt1 = hand_mark[i]
                x = int(pt1.x*width)
                # Procura: qual é o ponto mais a esquerda, qual o ponto mais a direita
                # qual é o ponto mais acima e qual o ponto mais abaixo respectivamente
                if(esq == None):
                    esq = x
                elif esq > x:
                     esq = x
                if(dir == None):
                    dir = x
                elif dir < x:
                    dir = x               
                y = int(pt1.y*height)
                if(cima == None):
                    cima = y
                elif cima < y:
                    cima = y
                if(baixo == None):
                    baixo = y
                elif baixo > y:
                    baixo = y
        if(cima != None):
            # caso uma mão tenha sido detectada, cria cubo na mão
            frame = cubo.criar_cubo(frame,[cima, baixo, dir, esq])
        cv.imshow('Detectar Mão', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()