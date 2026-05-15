import cv2
import numpy as np

camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)
dist_coeffs = np.array([0, 0, 0, 0], dtype=float)

# 2. Colocar o tamanho do marcador! Os que eu tô usando de teste tem esse tamanho, então no momento to deixando fisici
marker_size_cm = 5.7 # Physical size in cm
marker_size_m = marker_size_cm / 100
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #usa função pronta pra verificar os marcadores
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Estima local dos arucos
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size_m, camera_matrix, dist_coeffs)
        for i in range(len(ids)):
            #desenha o local deles, e adiciona eixos x,y,z neles!
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)
        if(len(ids) > 1):
            # Se tem mais de dois marcadores, faz o calculo da distância dos 2 primeiros
            c1 = corners[0][0]
            distAruco = cv2.norm(c1[0] - c1[1])
            center1 = (int(np.mean(c1[:, 0])), int(np.mean(c1[:, 1])))
            ratioCmPixel = distAruco/marker_size_cm
            c2 = corners[1][0]
            center2 = (int(np.mean(c2[:, 0])), int(np.mean(c2[:, 1])))
            #Cria uma linha entre os arucos, e pegue a mediana dessa linha! Nesse caso a mediana está um pouco acima no eixo y
            # pro texto não sobrepor a linha
            median = (int((center1[0] + center2[0]) / 2), (int((center1[1] + center2[1]) / 2)) + 10)
            
            cv2.line(frame, center1, center2, (184,90,193), 3)
            # Calcula a distância entre os dois pontos!
            dist = cv2.norm(tvecs[0] - tvecs[1])
            dist = dist*ratioCmPixel
            # Coloca o texto em cima da linha que conecta os dois pontos!
            frame = cv2.putText(frame, f"{dist:.4f} cm", median, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Câmera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
