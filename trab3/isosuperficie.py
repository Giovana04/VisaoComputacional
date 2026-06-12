from scipy.ndimage import gaussian_filter
import pyvista as pv

def isosuperficie(volume, valor_sigma):
    print("Gerando isosuperficie...")
    #volume = gaussian_filter(volume, sigma=2)
    volume_suavizado = gaussian_filter(volume, sigma=valor_sigma)

    grade = pv.ImageData()
    grade.dimensions = (volume_suavizado.shape[2], volume_suavizado.shape[1], volume_suavizado.shape[0])
    grade.spacing = (1.0, 1.0, 1.0)
    grade.origin = (0.0, 0.0, 0.0)
    
    grade.point_data["intensidade"] = volume_suavizado.flatten(order="C")
    
    malha = grade.contour(isosurfaces=[40])
    
    # aplicação decimação
    malha = malha.decimate(target_reduction=0.4)
    
    # p = pv.Plotter()
    # p.add_mesh(malha, color="tan", show_edges=False)
    
    # p.add_axes()
    # p.show()
    return malha, volume_suavizado