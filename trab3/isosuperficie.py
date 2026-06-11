from scipy.ndimage import gaussian_filter
import pyvista as pv
def isosuperficie(volume):
    print("Gerando isosuperficie...")
    # aplicação suavização
    volume = gaussian_filter(volume, sigma=1.4)
    
    grade = pv.ImageData()
    grade.dimensions = (volume.shape[2], volume.shape[1], volume.shape[0])
    grade.spacing = (1.0,1.0,1.0)
    grade.origin = (0.0,0.0,0.0)
    
    grade.point_data["intensidade"] = volume.flatten(order="C")
    
    malha = grade.contour(isosurfaces=[50])
    # aplicação decimação
    malha = malha.decimate(target_reduction=0.4)
    
    p = pv.Plotter()
    p.add_mesh(malha, color="tan", show_edges=False)
    
    p.add_axes()
    p.show()
    return malha, volume