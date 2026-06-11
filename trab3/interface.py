import tkinter as tk
from tkinter import ttk, messagebox

# Importa as suas funções mágicas
from calculo_volume import calculo_volume
from isosuperficie import isosuperficie
from esqueleto import esqueleto
from metricas import calcular_metricas
from visualizacao_dividida import janela_dividida

dados = {
    "volume": None,
    "superficie": None
}

def carregar_volume():
    pasta = combo_raizes.get()
    if not pasta:
        messagebox.showwarning("Aviso", "Escolha uma raiz primeiro")
        return
    
    lbl_status.config(text="Carregando volume...", fg="blue")
    janela.update()
    
    try:
        caminho = f'trab3\\{pasta}' 
        dados["volume"] = calculo_volume(caminho)
        lbl_status.config(text=f"Volume de {pasta} carregado com sucesso!", fg="green")
    except Exception as e:
        messagebox.showerror("Erro", f"volume:\n{e}")
        lbl_status.config(text="Erro ao carregar volume.", fg="red")

def gerar_iso():
    if dados["volume"] is None:
        messagebox.showwarning("Aviso", "Gere o volume primeiro.")
        return
    
    lbl_status.config(text="Gerando isosuperfície...", fg="blue")
    janela.update()
    
    try:
        dados["superficie"], dados["volume"] = isosuperficie(dados["volume"])
        lbl_status.config(text="Isosuperfície gerada!", fg="green")
    except Exception as e:
        messagebox.showerror("Erro", f"isosuperfície:\n{e}")

def gerar_esq():
    if dados["superficie"] is None:
        messagebox.showwarning("Aviso", "Gere a isosuperfície.")
        return
    try:
        esqueleto(dados["superficie"], dados["volume"])
    except Exception as e:
        messagebox.showerror("Erro", f"esqueleto:\n{e}")

def calcular_met():
    if dados["superficie"] is None:
        messagebox.showwarning("Aviso", "Gere a isosuperfície.")
        return
    
    lbl_status.config(text="Calculando métricas...", fg="blue")
    janela.update()
    
    try:
        # Agora recebe o texto formatado direto da função
        texto_resultado = calcular_metricas(dados["superficie"], dados["volume"])
        
        text_metricas.delete("1.0", tk.END)
        text_metricas.insert(tk.END, texto_resultado)
        lbl_status.config(text="Métricas calculadas e exibidas!", fg="green")
    except Exception as e:
        messagebox.showerror("Erro", f"métricas:\n{e}")

def mostrar_dividida():
    if dados["superficie"] is None:
        messagebox.showwarning("Aviso", "Gere a isosuperfície primeiro, senão uma das telas vai ficar vazia.")
        return
    try:
        janela_dividida(dados["volume"], dados["superficie"])
    except Exception as e:
        messagebox.showerror("Erro", f"visualização dividida:\n{e}")


# --- Configuração da Janela Principal ---
janela = tk.Tk()
janela.title("Trab 3 - Visão Computacional")
janela.geometry("450x650") # Dei uma esticada na janela para caber o texto
janela.eval('tk::PlaceWindow . center')

tk.Label(janela, text="Selecione a Raiz:", font=("Arial", 12)).pack(pady=(20, 5))

combo_raizes = ttk.Combobox(janela, values=["b0207", "b0309"], state="readonly", font=("Arial", 12))
combo_raizes.pack(pady=5)
combo_raizes.set("b0207")

tk.Button(janela, text="1. Carregar Volume (DVR)", command=carregar_volume, width=30, height=2).pack(pady=5)
tk.Button(janela, text="2. Gerar Isosuperfície", command=gerar_iso, width=30, height=2).pack(pady=5)
tk.Button(janela, text="3. Gerar Esqueleto", command=gerar_esq, width=30, height=2).pack(pady=5)
tk.Button(janela, text="4. Calcular Métricas", command=calcular_met, width=30, height=2).pack(pady=5)
tk.Button(janela, text="5. Visualização Dividida", command=mostrar_dividida, width=30, height=2).pack(pady=5)

lbl_status = tk.Label(janela, text="Aguardando comandos...", font=("Arial", 10, "italic"))
lbl_status.pack(pady=10)

# Aumentei o height para 13 para caber as 9 métricas + títulos sem scrollbar
text_metricas = tk.Text(janela, height=13, width=45, font=("Consolas", 10))
text_metricas.pack(pady=5)

janela.mainloop()