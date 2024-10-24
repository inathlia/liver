import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Menu
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from PIL import Image, ImageTk
import scipy.io

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador de Imagens")
        
        # Configuração do menu
        menu_bar = Menu(root)
        root.config(menu=menu_bar)
        
        # Menus
        file_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Arquivo", menu=file_menu)
        file_menu.add_command(label="Abrir Imagem", command=self.open_image)
        file_menu.add_command(label="Abrir Arquivo .MAT", command=self.open_mat)
        
        roi_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="ROI", menu=roi_menu)
        roi_menu.add_command(label="Selecionar ROI", command=self.select_roi)
        roi_menu.add_command(label="Exibir ROIs e Histogramas", command=self.display_rois)
        
        texture_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Textura", menu=texture_menu)
        texture_menu.add_command(label="Calcular GLCM", command=self.compute_glcm)
        texture_menu.add_command(label="Classificar Imagem", command=self.classify_image)

        # Área da imagem
        self.canvas = tk.Label(root)
        self.canvas.pack()
        
        # Inicializando variáveis
        self.image = None
        self.roi = None
        self.glcm = None

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)
            self.show_histogram(self.image)

    def open_mat(self):
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if file_path:
            mat_data = scipy.io.loadmat(file_path)
            if 'data' in mat_data:
                self.data_array = mat_data['data']
                self.images = self.data_array['images']
                # for n in range(55):
                #     for m in range(10):
                #         self.display_image_and_histogram(self.images[0][n][m])
            else:
                messagebox.showerror("Erro", "Arquivo MAT não contém um dado válido.")

    def display_image_and_histogram(self, img):
        # Create a figure with 2 subplots (1 row, 2 columns)
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the figsize to make space for both plots

        # Display the image on the left subplot (ax[0])
        axs[0].imshow(img, cmap='gray')  # Use 'gray' for grayscale images
        axs[0].axis('off')  # Hide axes for better visualization
        axs[0].set_title('Image')  # Optional: Set a title for the image

        # Display the histogram on the right subplot (ax[1])
        axs[1].hist(img.ravel(), bins=256, range=[0, 256])  # Flatten the image array and create the histogram
        axs[1].set_title('Histogram')  # Set a title for the histogram
        axs[1].set_xlabel('Pixel Intensity')  # Optional: Label for the x-axis
        axs[1].set_ylabel('Frequency')  # Optional: Label for the y-axis

        # Show the combined figure
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def select_roi(self):
        if self.image is None:
            messagebox.showwarning("Aviso", "Por favor, abra uma imagem primeiro.")
            return
        
        r = cv2.selectROI("Selecione a ROI", self.image)
        self.roi = self.image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        cv2.destroyAllWindows()
        self.display_image(self.roi)
        self.show_histogram(self.roi)

    def display_rois(self):
        if self.roi is None:
            messagebox.showwarning("Aviso", "Nenhuma ROI foi selecionada.")
            return
        self.display_image(self.roi)
        self.show_histogram(self.roi)
    
    def compute_glcm(self):
        if self.roi is None:
            messagebox.showwarning("Aviso", "Por favor, selecione uma ROI primeiro.")
            return

        distances = [1]
        angles = [0]
        levels = 256
        self.glcm = greycomatrix(self.roi, distances, angles, levels=levels, symmetric=True, normed=True)
        
        contrast = greycoprops(self.glcm, 'contrast')[0, 0]
        homogeneity = greycoprops(self.glcm, 'homogeneity')[0, 0]
        energy = greycoprops(self.glcm, 'energy')[0, 0]
        correlation = greycoprops(self.glcm, 'correlation')[0, 0]

        messagebox.showinfo("GLCM", f"Contraste: {contrast}\nHomogeneidade: {homogeneity}\nEnergia: {energy}\nCorrelação: {correlation}")
    
    def classify_image(self):
        # Esta função deverá ser expandida de acordo com o classificador escolhido pelo grupo
        if self.roi is None:
            messagebox.showwarning("Aviso", "Por favor, selecione uma ROI primeiro.")
            return

        # Aqui, você pode inserir um classificador baseado nos descritores de textura.
        messagebox.showinfo("Classificação", "A função de classificação será implementada.")

# Inicializando a aplicação
# if __name__ == "__main__":
    # root = tk.Tk()
    # app = ImageApp(root)
    # root.geometry("600x600")
    # root.mainloop()

