import NnetLZH
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # REQUIRED for embedding

class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Real-Time C++ Engine")

        # === 1. Load Model ===
        structure = [784, 128, 64, 10]
        self.model = NnetLZH.NeuralNet(structure)
        try:
            NnetLZH.LoadModel(self.model, "model_v1.nnet")
            print("Engine Online: Model Loaded")
        except:
            NnetLZH.RandomInitialise(self.model)
            print("Engine Online: Random Initialization")

        # === 2. GUI Layout (Two Columns) ===
        self.main_frame = tk.Frame(root)
        self.main_frame.pack()

        # Left Column: Controls
        self.left_col = tk.Frame(self.main_frame)
        self.left_col.pack(side=tk.LEFT, padx=10, pady=10)

        # Right Column: Live Brain View
        self.right_col = tk.Frame(self.main_frame)
        self.right_col.pack(side=tk.RIGHT, padx=10, pady=10)

        # --- Setup Canvas (Left) ---
        self.canvas_width, self.canvas_height = 280, 280
        self.canvas = tk.Canvas(self.left_col, width=self.canvas_width, height=self.canvas_height, bg="white", bd=2, relief="sunken")
        self.canvas.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw_handle = ImageDraw.Draw(self.image)
        
        # BINDINGS: Predict while drawing for real-time feel
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_draw)

        # --- Setup Matplotlib (Right) ---
        self.setup_live_brain()

        # Labels & Buttons
        self.prediction_label = tk.Label(self.left_col, text="Prediction: None", font=("Helvetica", 16))
        self.prediction_label.pack(pady=5)
        
        self.confidence_label = tk.Label(self.left_col, text="Confidence: N/A", font=("Helvetica", 10))
        self.confidence_label.pack(pady=5)
        
        tk.Button(self.left_col, text="Clear Canvas", command=self.clear_canvas).pack(pady=5)

    def setup_live_brain(self):
        """Initializes the embedded Matplotlib figure"""
        # Create a figure that fits in the GUI
        self.fig, self.axes = plt.subplots(1, len(self.model.layers), figsize=(5, 4))
        self.fig.patch.set_facecolor('#f0f0f0') 
        
        # Connect Matplotlib to Tkinter
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.right_col)
        self.plot_canvas.get_tk_widget().pack()
        self.update_brain_view()

    def update_brain_view(self):
        """Refreshes the heatmaps inside the GUI"""
        layers = self.model.layers
        for i, activations in enumerate(layers):
            self.axes[i].clear()
            act_array = np.array(activations).reshape(-1, 1)
            # 'magma' or 'viridis' work great for activations
            self.axes[i].imshow(act_array, cmap='magma', aspect='auto', vmin=0, vmax=1)
            self.axes[i].axis('off')
            self.axes[i].set_title(f"L{i}", fontsize=8)
        
        self.fig.tight_layout()
        self.plot_canvas.draw()

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        brush_size = 15
        if self.last_x and self.last_y:
            self.canvas.create_line((self.last_x, self.last_y, event.x, event.y), fill="black", width=brush_size, capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw_handle.line((self.last_x, self.last_y, event.x, event.y), fill=0, width=brush_size)
            # TRIGGER THE ENGINE!
            self.predict_digit()
        self.last_x, self.last_y = event.x, event.y

    def reset_draw(self, event):
        self.last_x, self.last_y = None, None

    def predict_digit(self):
        # Resize and Process
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_inverted = ImageOps.invert(img_resized)
        img_array = np.array(img_inverted, dtype=np.float32)
        cpp_input_vector = (img_array / 255.0).flatten().tolist()

        # Send to C++ engine
        self.model.input(cpp_input_vector)
        NnetLZH.FeedPropagation(self.model)
        
        # Update Labels
        res = self.model.MNISTResult()
        raw = self.model.rawResult()
        self.prediction_label.config(text=f"Prediction: {res}")
        self.confidence_label.config(text=f"Confidence: {raw[res]*100:.2f}%")
        
        # Update Heatmap
        self.update_brain_view()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw_handle = ImageDraw.Draw(self.image)
        self.predict_digit() # Reset brain view to zero

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawer(root)
    root.resizable(False, False)
    root.mainloop()
