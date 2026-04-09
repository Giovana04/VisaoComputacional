import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from combinacoes import criar_panorama as costurar
from gestos import iniciar, loop_gestos
caminho_img1 = ""
caminho_img2 = ""

def carregar_img1():
    global caminho_img1
    caminho_img1 = filedialog.askopenfilename(title="Selecione a Imagem Esquerda", filetypes=[("Imagens", "*.jpg *.png *.jpeg")])
    if caminho_img1:
        lbl_img1.config(text=f"Img 1: ...{caminho_img1[-20:]}")

def carregar_img2():
    global caminho_img2
    caminho_img2 = filedialog.askopenfilename(title="Selecione a Imagem Direita", filetypes=[("Imagens", "*.jpg *.png *.jpeg")])
    if caminho_img2:
        lbl_img2.config(text=f"Img 2: ...{caminho_img2[-20:]}")

def executar_panorama():
    if not caminho_img1 or not caminho_img2:
        messagebox.showerror("Erro", "Carregue as DUAS imagens")
        return

    combinacoes = [('ORB', 'BF'), ('ORB', 'FLANN'), ('SIFT', 'BF'), ('SIFT', 'FLANN')]
    resultados = []

    lbl_status.config(text="Processando...", fg="blue")
    janela.update()

    
    # Chama a função que está no outro arquivo
    for extrator, matcher in combinacoes:
        img, tempo = costurar(caminho_img1, caminho_img2, extrator, matcher)
        if img is not None:
            resultados.append((img, tempo, f"{extrator} + {matcher}"))

    lbl_status.config(text="Feche a janela de imagens para continuar", fg="green")

    if resultados:
        fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Costura de Imagens', fontsize=16)
        eixos = eixos.flatten()
        for i, (img, tempo, titulo) in enumerate(resultados):
            eixos[i].imshow(img)
            eixos[i].set_title(f"{titulo}\nTempo: {tempo:.4f} s", fontsize=12)
            eixos[i].axis('off')
        plt.show()
    else:
        messagebox.showwarning("Errpr", "Nenhuma das combinações gerou bons matches")
        lbl_status.config(text="fracasso.", fg="red")

def abrir_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Erro", "sem camera")
        return
    
    messagebox.showinfo("Aviso", "Aperte 'Q' na janela da câmera para fechar")
    iniciar(cap)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = loop_gestos(cap, ret, frame)
        cv2.putText(frame, "Aperte Q pra sair", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Interface Gestual', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

janela = tk.Tk()
janela.title("Visão computacional 1")
janela.geometry("420x360")
janela.eval('tk::PlaceWindow . center')

tk.Label(janela, text="Gerador de Panorama", font=("Arial", 16, "bold")).pack(pady=10)

btn_img1 = tk.Button(janela, text="1. Carregar Imagem", command=carregar_img1, width=25)
btn_img1.pack(pady=5)
lbl_img1 = tk.Label(janela, text="Img 1: Nenhuma", fg="gray")
lbl_img1.pack()

btn_img2 = tk.Button(janela, text="2. Carregar Imagem", command=carregar_img2, width=25)
btn_img2.pack(pady=5)
lbl_img2 = tk.Label(janela, text="Img 2: Nenhuma", fg="gray")
lbl_img2.pack()

tk.Frame(janela, height=2, bd=1, relief="sunken").pack(fill="x", padx=20, pady=10)

btn_gerar = tk.Button(janela, text="3. Gerar Panorama", command=executar_panorama, bg="lightblue", font=("Arial", 10, "bold"), width=25)
btn_gerar.pack(pady=5)

lbl_status = tk.Label(janela, text="", font=("Arial", 9))
lbl_status.pack()

tk.Frame(janela, height=2, bd=1, relief="sunken").pack(fill="x", padx=20, pady=5)

btn_camera = tk.Button(janela, text="Acessar Câmera (Gestos)", command=abrir_camera, bg="lightgray", width=25)
btn_camera.pack(pady=10)

janela.mainloop()