import os
import glob
from PIL import Image
import numpy as np
import cv2
import pyvista as pv
def remover_bordas(image):
    # converte a foto de Pillow pra Cv2
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    altura, largura = img.shape[:2]

    alt_cortada, lar_cortada = int(altura/1.4), int(largura/1.4)
    ini_x, ini_y = int((largura - lar_cortada)/2), int((altura-alt_cortada)/2)
    fim_x, fim_y = (ini_x+lar_cortada), (ini_y+alt_cortada)
    
    # remover bordas pegando só o centro da imagem
    centro_corte = img[ini_y:fim_y, ini_x:fim_x]
    
    #converte de cv2 pra Pillow
    return Image.fromarray(centro_corte)

def pre_processamento(img):
    img = remover_bordas(img)
    return img

def carregar_fotos_volume(pasta):
    # encontra todos os arquivos .png da pasta!
    caminho = os.path.join(pasta, "*.png")
    arquivos = sorted(glob.glob(caminho))
    with Image.open(arquivos[0]) as primeira_img:
        primeira_img = pre_processamento(primeira_img)
        img_cinza = primeira_img.convert("L")
        largura, altura = img_cinza.size
    profundidade = len(arquivos)
    array_volume = np.zeros((profundidade, altura, largura), dtype=np.uint8)
    
    for z, caminho_arquivo in enumerate(arquivos):
        with Image.open(caminho_arquivo) as img:
            img = pre_processamento(img)
            array_volume[z,:,:]  = np.array(img.convert("L"))
    return array_volume

def calculo_volume(pasta):
    print("Gerando Volume...")
    # fala qual conjunto de fotos vai ser lido
    pasta = pasta
    volume = carregar_fotos_volume(pasta)
    
    grid = pv.ImageData()
    
    grid.dimensions = (volume.shape[2], volume.shape[1], volume.shape[0])
    
    grid.point_data["densidade"] = volume.flatten(order="C")
    
    plotter = pv.Plotter()
    plotter.add_volume(
        grid,
        scalars="densidade",
        cmap="grayC",
        opacity="sigmoid",
        blending="composite"
    )
    plotter.show()
    return volume