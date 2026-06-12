from skimage.morphology import skeletonize
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv

def esqueleto(volume):
    print('Gerando Esqueleto....')
    volume = volume > 50
    esqueleto = skeletonize(volume) 
    coordenadas = np.argwhere(esqueleto).astype(np.float32)
    coordenadas = coordenadas[:, [2, 1, 0]]
    arvore = KDTree(coordenadas)
    
    pares = arvore.query_pairs(r=1.8) 
    
    linhas_pv = []
    for ind_a, ind_b in pares:
        linhas_pv.extend([2, ind_a, ind_b])
    
    grafo = pv.PolyData(coordenadas)
    grafo.lines = np.array(linhas_pv)
    
    # p = pv.Plotter()
    # p.add_mesh(isosuperficie, color="tan", opacity=0.4, show_edges=False)
    
    # tubos = grafo.tube(radius=1)
    # p.add_mesh(tubos, color="blue", smooth_shading=True)
    
    # p.add_axes()
    # p.show()

    return grafo.tube(radius=1)