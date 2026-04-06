import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def criar_panorama(img_esq_path, img_dir_path, extrator_tipo, matcher_tipo):
    inicio_tempo = time.time()

    img_esq = cv2.imdecode(np.fromfile(img_esq_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_dir = cv2.imdecode(np.fromfile(img_dir_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    if img_esq is None or img_dir is None:
        print("Imagem não encontrada")
        return None, 0

    # SIFT e ORB trabalham com imagens em tons de cinza
    gray_esq = cv2.cvtColor(img_esq, cv2.COLOR_BGR2GRAY)
    gray_dir = cv2.cvtColor(img_dir, cv2.COLOR_BGR2GRAY)

    # 1. Encontrar pontos de interesse
    if extrator_tipo == 'SIFT':
        detector = cv2.SIFT_create() 
        norm_type = cv2.NORM_L2
    elif extrator_tipo == 'ORB':
        detector = cv2.ORB_create(nfeatures=2000) 
        norm_type = cv2.NORM_HAMMING
    
    kp1, des1 = detector.detectAndCompute(gray_esq, None)
    kp2, des2 = detector.detectAndCompute(gray_dir, None)

    # 2. Encontrar correspondências
    matches_bons = []
    
    if matcher_tipo == 'BF':
        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                matches_bons.append(m)

    elif matcher_tipo == 'FLANN':
        if extrator_tipo == 'SIFT':
            index_params = dict(algorithm=1, trees=5) 
        else:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1) 
        
        search_params = dict(checks=50) 
        flann = cv2.FlannBasedMatcher(index_params, search_params) 
        matches = flann.knnMatch(des1, des2, k=2) 
             
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    matches_bons.append(m)

    # 3. Homografia e Warping 
    MIN_MATCH_COUNT = 10 
    if len(matches_bons) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches_bons]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches_bons]).reshape(-1, 1, 2)

        M, mascara = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h1, w1 = img_esq.shape[:2]
        h2, w2 = img_dir.shape[:2]

        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        pts1_warped = cv2.perspectiveTransform(pts1, M)
        pts = np.concatenate((pts2, pts1_warped), axis=0)

        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

        resultado = cv2.warpPerspective(img_esq, Ht.dot(M), (xmax - xmin, ymax - ymin))
        resultado[t[1]:h2+t[1], t[0]:w2+t[0]] = img_dir

        tempo_total = time.time() - inicio_tempo
        
        # Converter BGR para RGB para o Matplotlib não estragar as cores
        resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
        return resultado_rgb, tempo_total
        
    else:
        print(f"[{extrator_tipo}+{matcher_tipo}] Só {len(matches_bons)} correspondências.")
        return None, 0
