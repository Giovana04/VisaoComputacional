import cv2
import numpy as np
import pygame

def desenhar_ocarina_codigo(img, rvec, tvec, mtx, dist, estado_furos):
    escala = 0.01
    comp, larg, alt = 6.0 * escala, 2.0 * escala, 1.5 * escala    
    
    vertices = np.float32([
        [0, 0, alt], [0, comp, 0], [larg, larg, 0], [-larg, larg, 0], [0, -comp/2, larg], [0, 0, -alt/2]
    ])
    imgpts, _ = cv2.projectPoints(vertices, rvec, tvec, mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    cor_face = (180, 70, 40) 
    cor_borda = (255, 200, 100) # borda clara
    esp = 2

    faces = [(0,1,2), (0,1,3), (0,4,2), (0,4,3)]
    for face in faces:
        pts = np.array([imgpts[face[0]], imgpts[face[1]], imgpts[face[2]]])
        # pinta
        cv2.fillConvexPoly(img, pts, cor_face)
        # desenha a borda por cima
        cv2.polylines(img, [pts], True, cor_borda, esp)

    furos_3d = {
        18: [larg/1.5, comp/4, alt/1.5],   # Furo 1
        12: [-larg/1.5, comp/4, alt/1.5],  # Furo 2 
        27: [larg/2, comp/1.5, alt/3],     # Furo 3
        43: [-larg/2, comp/1.5, alt/3],    # Furo 4
        5:  [0, comp/2, alt/1.2]           # Furo 5 (Centro)
    }
    
    lista_ids = [18, 12, 27, 43, 5]
    pontos_furos = np.float32([furos_3d[i] for i in lista_ids])
    imgpts_furos, _ = cv2.projectPoints(pontos_furos, rvec, tvec, mtx, dist)
    imgpts_furos = np.int32(imgpts_furos).reshape(-1, 2)
    
    # vermelho se tocando, preto se aberto
    for idx, furo_id in enumerate(lista_ids):
        pt = tuple(imgpts_furos[idx])
        ta_tampado = estado_furos.get(furo_id, False)
        
        cor_furo = (0, 0, 255) if ta_tampado else (30, 30, 30) 
        cv2.circle(img, pt, 6, cor_furo, -1)
        cv2.circle(img, pt, 6, (255, 255, 255), 1)

    return img

def desenhar_hud(img, estado_furos):
    furos_ativos = [str(f_id) for f_id, ativo in estado_furos.items() if ativo]
    texto = f"Tocando Furos: {', '.join(furos_ativos)}" if furos_ativos else "Silencio..."
    
    cv2.putText(img, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(img, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

def rodar_ocarina():
    pygame.mixer.init()
    
    mapa_furos = {
        18: "sons/1.mp3",
        12: "sons/2.mp3",
        27: "sons/3.mp3",
        43: "sons/4.mp3",
        5:  "sons/5.mp3"
    }
    
    sons = {}
    for aruco_id, caminho_som in mapa_furos.items():
        try:
            sons[aruco_id] = pygame.mixer.Sound(caminho_som)
        except Exception:
            class Mudo: 
                def play(self, *args, **kwargs): pass
                def fadeout(self, *args, **kwargs): pass
                def stop(self): pass
            sons[aruco_id] = Mudo()

    camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=float)
    dist_coeffs = np.array([0, 0, 0, 0], dtype=float)
    marker_size_m = 0.034 

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()

    estado_furos = {18: False, 12: False, 27: False, 43: False, 5: False}
    furos_vistos = {18: False, 12: False, 27: False, 43: False, 5: False}
    
    ID_BASE = 42
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            ids_flat = ids.flatten()
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size_m, camera_matrix, dist_coeffs)
            
            if ID_BASE in ids_flat:
                idx_ref = np.where(ids_flat == ID_BASE)[0][0]
                
                furos_que_sumiram_neste_frame = []

                for furo_id in mapa_furos.keys():
                    if furo_id in ids_flat:
                        furos_vistos[furo_id] = True
                        if estado_furos[furo_id]:
                            sons[furo_id].fadeout(200) 
                            estado_furos[furo_id] = False
                    else:
                        if furos_vistos[furo_id] and not estado_furos[furo_id]:
                            furos_que_sumiram_neste_frame.append(furo_id)

                if len(furos_que_sumiram_neste_frame) > 0:
                    if len(furos_que_sumiram_neste_frame) > 2:
                        furos_pra_tocar = [furos_que_sumiram_neste_frame[0]]
                    else:
                        furos_pra_tocar = furos_que_sumiram_neste_frame
                    
                    for f_id in furos_pra_tocar:
                        sons[f_id].play(loops=-1, fade_ms=50)
                    
                    for f_id in furos_que_sumiram_neste_frame:
                        estado_furos[f_id] = True

                desenhar_ocarina_codigo(frame, rvecs[idx_ref], tvecs[idx_ref], camera_matrix, dist_coeffs, estado_furos)

            else:
                for furo_id in mapa_furos.keys():
                    if estado_furos[furo_id]:
                        sons[furo_id].stop()
                estado_furos = {18: False, 12: False, 27: False, 43: False, 5: False}
                furos_vistos = {18: False, 12: False, 27: False, 43: False, 5: False}

                cv2.aruco.drawDetectedMarkers(frame, corners) # desenha os quadrado verde

        else:
            for furo_id in mapa_furos.keys():
                if estado_furos[furo_id]:
                    sons[furo_id].stop()
            estado_furos = {18: False, 12: False, 27: False, 43: False, 5: False}
            furos_vistos = {18: False, 12: False, 27: False, 43: False, 5: False}
        
        desenhar_hud(frame, estado_furos)
        
        cv2.imshow('Ocarina', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rodar_ocarina()