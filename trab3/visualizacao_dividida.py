import pyvista as pv

def janela_dividida(volume_array, malha):
    print("Gerando visualização dividida")
    
    p = pv.Plotter(shape=(1, 2))
    
    p.subplot(0, 0)
    p.add_text("DVR", font_size=14)
    grid = pv.ImageData()
    grid.dimensions = (volume_array.shape[2], volume_array.shape[1], volume_array.shape[0])
    grid.point_data["densidade"] = volume_array.flatten(order="C")
    p.add_volume(
        grid, scalars="densidade", cmap="grayC", 
        opacity="sigmoid", blending="composite"
    )
    
    p.subplot(0, 1)
    p.add_text("Isosuperfície", font_size=14)
    p.add_mesh(malha, color="tan", show_edges=False)
    p.add_axes()
    
    # Linca as câmeras para o movimento síncrono e exibe
    p.link_views()
    p.show()