import tkinter as tk
from tkinter import ttk, messagebox
import pyvista as pv

from calculo_volume import calculo_volume
from isosuperficie import isosuperficie
from esqueleto import esqueleto
from metricas import calcular_metricas

# Dicionário mestre para segurar tudo
dados = {
    "b0207": {"volume": None, "superficie": None, "tubos": None, "sigma": 2.2},
    "b0309": {"volume": None, "superficie": None, "tubos": None, "sigma": 1.2}
}

def carregar_volumes():
    lbl_status.config(text="Carregando matrizes...", foreground="blue")
    janela.update()
    try:
        p = pv.Plotter(shape=(1, 2))
        for i, raiz in enumerate(["b0207", "b0309"]):
            dados[raiz]["volume"] = calculo_volume(f'trab3\\{raiz}')
            
            p.subplot(0, i)
            p.add_text(f"DVR - {raiz}", font_size=12)
            grid = pv.ImageData()
            vol = dados[raiz]["volume"]
            grid.dimensions = (vol.shape[2], vol.shape[1], vol.shape[0])
            grid.point_data["densidade"] = vol.flatten(order="C")
            p.add_volume(grid, scalars="densidade", cmap="gray", opacity="sigmoid", blending="composite")
            
        lbl_status.config(text="Volumes carregados na memória!", foreground="green")
        p.link_views()
        p.show()
    except Exception as e:
        messagebox.showerror("Erro", f"Volume falhou:\n{e}")

def gerar_isos():
    if dados["b0207"]["volume"] is None:
        messagebox.showwarning("Aviso", "Aperte o botão 1 primeiro.")
        return
    lbl_status.config(text="Lapidando malhas 3D...", foreground="blue")
    janela.update()
    try:
        p = pv.Plotter(shape=(1, 2))
        for i, raiz in enumerate(["b0207", "b0309"]):
            malha, vol_suavizado = isosuperficie(dados[raiz]["volume"], dados[raiz]["sigma"])
            dados[raiz]["superficie"] = malha
            dados[raiz]["volume"] = vol_suavizado
            
            p.subplot(0, i)
            p.add_text(f"Isosuperfície {raiz} (Sigma: {dados[raiz]['sigma']})", font_size=12)
            p.add_mesh(malha, color="tan", show_edges=False)
            p.add_axes()
            
        lbl_status.config(text="Isosuperfícies prontas!", foreground="green")
        p.link_views()
        p.show()
    except Exception as e:
        messagebox.showerror("Erro", f"Isosuperfície:\n{e}")

def gerar_esq():
    if dados["b0207"]["superficie"] is None:
        messagebox.showwarning("Aviso", "Sem isosuperfície.")
        return
    lbl_status.config(text="Extraindo...", foreground="blue")
    janela.update()
    try:
        p = pv.Plotter(shape=(1, 2))
        for i, raiz in enumerate(["b0207", "b0309"]):
            tubos = esqueleto(dados[raiz]["volume"])
            dados[raiz]["tubos"] = tubos
            
            p.subplot(0, i)
            p.add_text(f"Esqueleto - {raiz}", font_size=12)
            p.add_mesh(dados[raiz]["superficie"], color="tan", opacity=0.3, show_edges=False)
            p.add_mesh(tubos, color="blue", smooth_shading=True)
            p.add_axes()
            
        lbl_status.config(text="Esqueletos gerados!", foreground="green")
        p.link_views()
        p.show()
    except Exception as e:
        messagebox.showerror("Erro", f"esqueleto:\n{e}")

def calcular_met():
    if dados["b0207"]["superficie"] is None:
        messagebox.showwarning("Aviso", "Matemática requer malhas existentes.")
        return
    lbl_status.config(text="Calculando...", foreground="blue")
    janela.update()
    try:
        txt_0207.delete("1.0", tk.END)
        txt_0207.insert(tk.END, f"RAIZ b0207 (Sigma 2.2)\n{calcular_metricas(dados['b0207']['superficie'], dados['b0207']['volume'])}")
        
        txt_0309.delete("1.0", tk.END)
        txt_0309.insert(tk.END, f"RAIZ b0309 (Sigma 1.2)\n{calcular_metricas(dados['b0309']['superficie'], dados['b0309']['volume'])}")
        
        lbl_status.config(text="Métricas calculadas!", foreground="green")
    except Exception as e:
        messagebox.showerror("Erro", f"A matemática falhou:\n{e}")

def mostrar_dividida():
    if dados["b0207"]["superficie"] is None:
        messagebox.showwarning("Aviso", "Siga a ordem dos botões.")
        return
    try:
        p = pv.Plotter(shape=(2, 2))
        for i, raiz in enumerate(["b0207", "b0309"]):
            vol = dados[raiz]["volume"]
            sup = dados[raiz]["superficie"]
            
            p.subplot(i, 0)
            p.add_text(f"DVR - {raiz}", font_size=10)
            grid = pv.ImageData()
            grid.dimensions = (vol.shape[2], vol.shape[1], vol.shape[0])
            grid.point_data["densidade"] = vol.flatten(order="C")
            p.add_volume(grid, scalars="densidade", cmap="gray", opacity="sigmoid", blending="composite")
            
            p.subplot(i, 1)
            p.add_text(f"Iso - {raiz}", font_size=10)
            p.add_mesh(sup, color="tan", show_edges=False)
            
        p.link_views()
        p.show()
    except Exception as e:
        messagebox.showerror("Erro", f"Visão dividida:\n{e}")

janela = tk.Tk()
janela.title("Visão Computacional")
janela.geometry("680x600")
janela.eval('tk::PlaceWindow . center')

style = ttk.Style(janela)
style.theme_use('clam')

titulo = tk.Label(janela, text="MENU", font=("Segoe UI", 16, "bold"))
titulo.pack(pady=15)

# Frame central para os botões ficarem alinhados
frame_botoes = tk.Frame(janela)
frame_botoes.pack(pady=5)

ttk.Button(frame_botoes, text="1. Carregar Volumes", command=carregar_volumes, width=25).grid(row=0, column=0, padx=5, pady=5)
ttk.Button(frame_botoes, text="2. Gerar Isosuperfícies", command=gerar_isos, width=25).grid(row=0, column=1, padx=5, pady=5)
ttk.Button(frame_botoes, text="3. Gerar Esqueletos", command=gerar_esq, width=25).grid(row=1, column=0, padx=5, pady=5)
ttk.Button(frame_botoes, text="4. Calcular Métricas", command=calcular_met, width=25).grid(row=1, column=1, padx=5, pady=5)

ttk.Button(janela, text="5. Visualização Dividida", command=mostrar_dividida, width=53).pack(pady=10)

lbl_status = ttk.Label(janela, text="Aguardando comandos...", font=("Segoe UI", 10, "italic"))
lbl_status.pack(pady=5)

frame_textos = tk.Frame(janela)
frame_textos.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)

txt_0207 = tk.Text(frame_textos, width=38, height=14, font=("Consolas", 9), bg="#f4f4f4")
txt_0207.pack(side=tk.LEFT, padx=5)

txt_0309 = tk.Text(frame_textos, width=38, height=14, font=("Consolas", 9), bg="#f4f4f4")
txt_0309.pack(side=tk.RIGHT, padx=5)

janela.mainloop()