import tkinter as tk
from tkinter import ttk
import subprocess
import sys

def rodar_script(nome_arquivo):
    print(f"Invocando o script: {nome_arquivo}...")
    subprocess.Popen([sys.executable, nome_arquivo])

root = tk.Tk()
root.title("Trabalho 2")
root.geometry("400x350")
root.configure(bg="#1e1e1e") 
root.resizable(False, False)

style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', 
                font=('Segoe UI', 12, 'bold'), 
                padding=12, 
                background="#007acc", 
                foreground="white",
                borderwidth=0)
style.map('TButton', background=[('active', '#005999')]) 

lbl_titulo = tk.Label(root, 
                      text="Menu", 
                      font=('Segoe UI', 18, 'bold'), 
                      bg="#1e1e1e", 
                      fg="#ffffff")
lbl_titulo.pack(pady=(30, 10))

lbl_sub = tk.Label(root, 
                   text="Selecione o modo de execução:", 
                   font=('Segoe UI', 10), 
                   bg="#1e1e1e", 
                   fg="#aaaaaa")
lbl_sub.pack(pady=(0, 20))

btn_a = ttk.Button(root, text="A) Metrologia", command=lambda: rodar_script('conecta_arucos.py'))
btn_a.pack(fill='x', padx=60, pady=8)

btn_b = ttk.Button(root, text="B) Ocarina 3D", command=lambda: rodar_script('ocarina.py'))
btn_b.pack(fill='x', padx=60, pady=8)

btn_c = ttk.Button(root, text="C) VR Sem Marcadores", command=lambda: rodar_script('realidade_virtual2.py'))
btn_c.pack(fill='x', padx=60, pady=8)

root.mainloop()