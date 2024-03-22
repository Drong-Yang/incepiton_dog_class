"""GUI application for dog breed classification."""
import tkinter as tk
from tkinter import filedialog, ttk

class DogClassifierGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Dog Breed Classifier')
        self.root.geometry('800x600')
        self.setup_ui()
    def setup_ui(self):
        self.upload_btn = ttk.Button(self.root, text='Upload Image')
        self.upload_btn.pack(pady=20)
        self.result_label = ttk.Label(self.root, text='')
        self.result_label.pack()
    def run(self):
        self.root.mainloop()
