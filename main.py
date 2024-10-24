import os
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

class LiverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Liver Classifier")
        self.patient_num = tk.IntVar(value=0)
        self.image_num = tk.IntVar(value=0)
        self.max_patients = 54
        self.max_images = 9
        self.liver_dataset = 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'
        self.histogram_window = None 
        self.image = None

        # create main menu
        menu = tk.Menu(root)
        root.config(menu=menu)

        # upload file menu
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image (.png, .jpg)", command=self.load_image)
        file_menu.add_command(label="Load Image (.mat)", command=self.load_and_show_mat)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)

        # image menu
        image_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Image", menu=image_menu)
        image_menu.add_command(label="View Histogram", command=self.view_histogram)
        image_menu.add_command(label="Crop ROI", command=self.crop_roi)
        image_menu.add_command(label="View ROI & Histogram", command=self.view_roi_histogram)

        # analysis menu
        analysis_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Compute GLCM & Texture Descriptors", command=self.compute_glcm)
        analysis_menu.add_command(label="Characterize ROI", command=self.characterize_roi)
        analysis_menu.add_command(label="Classify Image", command=self.classify_image)

        self.create_selection_frame()
        self.create_navigation_buttons()
        self.create_direct_load_button()
        self.create_roi_button()

    # images navigation -------------------------------------------------------------------------
    def get_patient_number(self):
        return self.patient_num.get()

    def get_image_number(self):
        return self.image_num.get()

    def set_patient_number(self, n):
        self.patient_num.set(n)

    def set_image_number(self, m):
        self.image_num.set(m)

    def create_selection_frame(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        tk.Label(frame, text="Select patient (0-54):").grid(row=0, column=0, padx=5, pady=5)
        patient_spinbox = ttk.Spinbox(frame, from_=0, to=self.max_patients, textvariable=self.patient_num, width=5)
        patient_spinbox.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(frame, text="Select image (0-9):").grid(row=1, column=0, padx=5, pady=5)
        image_spinbox = ttk.Spinbox(frame, from_=0, to=self.max_images, textvariable=self.image_num, width=5)
        image_spinbox.grid(row=1, column=1, padx=5, pady=5)
    
    def create_navigation_buttons(self):
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=10)

        prev_button = tk.Button(nav_frame, text="Previous", command=self.previous_image)
        prev_button.grid(row=0, column=0, padx=10)

        next_button = tk.Button(nav_frame, text="Next", command=self.next_image)
        next_button.grid(row=0, column=1, padx=10)

    def update_image(self):
        n = self.get_patient_number()
        m = self.get_image_number()

        self.image = self.images[0][n][m]

        if len(self.image.shape) > 2:
            self.image = np.mean(self.image, axis=2)  # Converte RGB para grayscale

        self.show_image_and_histogram()

    def next_image(self):
        current_patient = self.get_patient_number()
        current_image = self.get_image_number()

        if current_image < self.max_images:
            self.set_image_number(current_image + 1)
        elif current_patient < self.max_patients:
            self.set_patient_number(current_patient + 1)
            self.set_image_number(0)

        self.update_image()

    def previous_image(self):
        current_patient = self.get_patient_number()
        current_image = self.get_image_number()

        if current_image > 0:
            self.set_image_number(current_image - 1)
        elif current_patient > 0:
            self.set_patient_number(current_patient - 1)
            self.set_image_number(self.max_images)

        self.update_image()

    def create_direct_load_button(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)

        load_button = tk.Button(button_frame, text="Liver dataset", command=self.load_and_show_liver)
        load_button.grid(row=0, column=0, padx=10)

    # upload file menu functions -------------------------------------------------------------------------
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)
            self.show_histogram(self.image)

    def load_and_show_liver(self):
        self.file_path = self.liver_dataset
        self.load_mat()
    
    def load_and_show_mat(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        self.load_mat()

    def load_mat(self):
        if self.file_path:
            data = scipy.io.loadmat(self.file_path)
            if 'data' in data:
                self.data_array = data['data']
                self.images = self.data_array['images']

                # values from selection frame
                n = self.get_patient_number()
                m = self.get_image_number()

                self.image = self.images[0][n][m]

                if len(self.image.shape) > 2:
                    self.image = np.mean(self.image, axis=2)  # Converte RGB para grayscale
                self.show_image_and_histogram()
                
            else:
                messagebox.showerror("Erro", ".mat file not valid.")

    def show_image_and_histogram(self):
        plt.close('all')

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title("Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.hist(self.image.ravel(), bins=256, range=(0, 256), color='black')
        plt.title("Grayscale Histogram")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.show()

    # image menu functions -----------------------------------------------------------------
    # roi functions
    def create_roi_button(self):
        roi_frame = tk.Frame(self.root)
        roi_frame.pack(pady=20)

        roi_button = tk.Button(roi_frame, text="Select ROI", command=self.select_roi)
        roi_button.grid(row=0, column=0, padx=10)

    def select_roi(self):
        if self.image is None:
            messagebox.showwarning("Warning", "No image loaded.")
            return

        plt.close('all')
        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap='gray')
        ax.set_title("Select ROI")
        plt.axis('on')

        def on_select(event):
            if event.button == 1:  # Botão esquerdo do mouse
                x0, y0 = event.xdata, event.ydata
                roi = plt.ginput(2)  # Pega dois pontos para definir a ROI
                if len(roi) == 2:
                    x1, y1 = map(int, roi[1])  # Segundo ponto
                    x0, y0 = map(int, roi[0])  # Primeiro ponto
                    self.roi = self.image[y0:y1, x0:x1]  # Corta a ROI
                    plt.close()  # Fecha a figura após seleção

                    # Exibe a ROI cortada
                    self.show_cropped_roi()

        # Conecta o evento de clique
        cid = fig.canvas.mpl_connect('button_press_event', on_select)
        plt.show()

    def show_cropped_roi(self):
        if self.roi is not None:
            plt.figure(figsize=(5, 5))
            plt.imshow(self.roi, cmap='gray')
            plt.title("ROI Cropped")
            plt.axis('off')
            plt.show()
        else:
            messagebox.showwarning("Warning", "No ROI loaded.")

    def crop_roi(self):
        # Function to crop a region of interest from the image (Placeholder)
        messagebox.showinfo("Crop ROI", "Select a region of interest to crop and save.")

    #  histogram functions
    def view_histogram(self):
        # Function to view histogram of the image (Placeholder)
        messagebox.showinfo("View Histogram", "Displaying histogram of the loaded image.")

    def view_roi_histogram(self):
        # Function to view ROI and its histogram (Placeholder)
        messagebox.showinfo("View ROI & Histogram", "Displaying ROI and its histogram.")

    # analysis menu functions ----------------------------------------------------------------
    def compute_glcm(self):
        # Function to compute GLCM and display texture descriptors (Placeholder)
        messagebox.showinfo("Compute GLCM", "Computing GLCM and texture descriptors for the ROI.")

    def characterize_roi(self):
        # Function to characterize ROI based on texture descriptor (Placeholder)
        messagebox.showinfo("Characterize ROI", "Characterizing the ROI with the selected texture descriptor.")

    def classify_image(self):
        # Function to classify the image based on selected technique (Placeholder)
        messagebox.showinfo("Classify Image", "Classifying the image and identifying its class.")

# create the main window
if __name__ == "__main__":
    root = tk.Tk()
    app = LiverApp(root)
    root.geometry("600x400")
    root.mainloop()