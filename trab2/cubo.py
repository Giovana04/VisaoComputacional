import cv2
def criar_cubo(img, coords):
    # as coordenadas da parte de traz do cubo são um pouco afastadas das originais!                
    coordBottom = [(int((coords[1]+coords[0])/10)+coords[0]),(int((coords[1]+coords[0])/10)+coords[1]), (int((coords[3]+coords[2])/10)+coords[2]), (int((coords[3]+coords[2])/10)+coords[3])]
    # Coordenadas dos pontos do retangulo da frente do cubo
    f_tl, f_tr = (coords[3], coords[0]), (coords[2], coords[0])
    f_bl, f_br = (coords[3], coords[1]), (coords[2], coords[1])
    # Coordenadas dos pontos do retangulo de traz do cubo
    b_tl, b_tr = (coordBottom[3], coordBottom[0]), (coordBottom[2], coordBottom[0])
    b_bl, b_br = (coordBottom[3], coordBottom[1]), (coordBottom[2], coordBottom[1])


    # Primeiro retangulo de traz
    cv2.rectangle(img, b_tl, b_br, (0, 255, 0), 2)

    # Conecta as faces
    lines = [(f_tl, b_tl), (f_tr, b_tr), (f_bl, b_bl), (f_br, b_br)]
    for start, end in lines:
        cv2.line(img, start, end, (255, 0, 0), 2)

    # Segundo retangulo
    cv2.rectangle(img, f_tl, f_br, (0, 0, 255), 2)
    return img