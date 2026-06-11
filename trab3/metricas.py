import numpy as np
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from scipy.ndimage import convolve

def calcular_metricas(malha, volume_array):
    vol = malha.volume
    area = malha.area
    
    compacidade = (area ** 1.5) / vol if vol > 0 else 0
    
    vol_binario = volume_array > 50
    props = regionprops(vol_binario.astype(int))[0]
    eixo_maior = props.axis_major_length
    eixo_menor = props.axis_minor_length
    excentricidade = np.sqrt(1 - (eixo_menor**2 / eixo_maior**2)) if eixo_maior > 0 else 0
    
    esqueleto_bin = skeletonize(vol_binario)
    
    kernel = np.ones((3, 3, 3))
    kernel[1, 1, 1] = 0
    vizinhos = convolve(esqueleto_bin.astype(int), kernel, mode='constant')
    vizinhos_no_esqueleto = vizinhos * esqueleto_bin
    
    voxeis_esqueleto = np.sum(esqueleto_bin)
    endpoints = np.sum(vizinhos_no_esqueleto == 1)
    bifurcacoes = np.sum(vizinhos_no_esqueleto > 2)
    caminhos_simples = np.sum(vizinhos_no_esqueleto == 2)
    densidade_esq = voxeis_esqueleto / np.sum(vol_binario)
    
    texto_resultado = (
        f"--- MÉTRICAS GERAIS ---\n"
        f"Volume: {vol:.2f}\n"
        f"Área de Superfície: {area:.2f}\n"
        f"Compacidade: {compacidade:.2f}\n"
        f"Excentricidade: {excentricidade:.2f}\n\n"
        f"--- MÉTRICAS DO ESQUELETO ---\n"
        f"1. Vóxeis do Esqueleto: {voxeis_esqueleto}\n"
        f"2. Endpoints (Pontas): {endpoints}\n"
        f"3. Bifurcações (Nós): {bifurcacoes}\n"
        f"4. Caminhos Simples: {caminhos_simples}\n"
        f"5. Densidade (Esq/Vol): {densidade_esq:.6f}"
    )
    
    return texto_resultado