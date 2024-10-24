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

        # create main menu
        menu = tk.Menu(root)
        root.config(menu=menu)

        # upload file menu
        file_menu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image (.png, .jpg)", command=self.load_image)
        file_menu.add_command(label="Load Image (.mat)", command=self.load_mat)
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

    # upload file menu functions -------------------------------------------------------------------------
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)
            self.show_histogram(self.image)

    def load_mat(self):
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if file_path:
            data = scipy.io.loadmat(file_path)
            if 'data' in data:
                self.data_array = data['data']
                self.images = self.data_array['images']

                # values from selection frame
                n = 1
                m = 5

                self.image = self.images[0][n][m]

                self.show_image_and_histogram()
                
            else:
                messagebox.showerror("Erro", ".mat file not valid.")

    def show_image_and_histogram(self):
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
    def view_histogram(self):
        # Function to view histogram of the image (Placeholder)
        messagebox.showinfo("View Histogram", "Displaying histogram of the loaded image.")

    def crop_roi(self):
        # Function to crop a region of interest from the image (Placeholder)
        messagebox.showinfo("Crop ROI", "Select a region of interest to crop and save.")

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