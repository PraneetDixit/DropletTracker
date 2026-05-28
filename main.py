import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import cinereader as cr
import os

from image_processor import detect_droplet
from data_exporter import export_and_plot_single, plot_multi_comparison

# ==========================================
# MODULE 1: VIDEO ANALYZER
# ==========================================
class VideoAnalyzerTab:
    def __init__(self, parent, root):
        self.parent = parent
        self.root = root
        
        self.images = None
        self.timestamps = None
        self.num_frames = 0
        self.fps = 1
        self.current_frame_idx = 0
        
        self.resize_factor = 1.0
        self.raw_baseline_y = 500  
        self.bounding_boxes = {} 
        
        self.interaction_mode = None 
        self.start_x = 0
        self.start_y = 0
        self.temp_box_cache = None 

        self.is_playing = False
        self.play_direction = 1
        self.playback_delay = 50 
        
        self.setup_ui()
        self.root.bind("<Escape>", self.clear_current_box)

    def setup_ui(self):
        control_frame = tk.Frame(self.parent, width=250, padx=15, pady=15, bg="#f0f0f0")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(control_frame, text="1. Load .CINE File", command=self.load_file, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), pady=5).pack(fill=tk.X, pady=10)
        
        tk.Label(control_frame, text="1 mm = (raw pixels):", bg="#f0f0f0").pack(anchor=tk.W, pady=(10,2))
        self.scale_entry = tk.Entry(control_frame)
        self.scale_entry.insert(0, "50") 
        self.scale_entry.pack(fill=tk.X)

        tk.Label(control_frame, text="Initial Dia (mm):", bg="#f0f0f0").pack(anchor=tk.W, pady=(10,2))
        self.init_dia_entry = tk.Entry(control_frame)
        self.init_dia_entry.insert(0, "2.52") 
        self.init_dia_entry.pack(fill=tk.X)

        tk.Label(control_frame, text="Impact Frame (t=0):", bg="#f0f0f0").pack(anchor=tk.W, pady=(15,2))
        self.t0_entry = tk.Entry(control_frame)
        self.t0_entry.insert(0, "0")
        self.t0_entry.pack(fill=tk.X)
        tk.Button(control_frame, text="Set Current Frame as t=0", command=self.set_t0).pack(fill=tk.X, pady=5)

        nav_frame = tk.Frame(control_frame, bg="#f0f0f0")
        nav_frame.pack(pady=(15, 2), fill=tk.X)
        tk.Button(nav_frame, text="< Prev", command=self.manual_prev).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(nav_frame, text="Next >", command=self.manual_next).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=2)

        playback_frame = tk.Frame(control_frame, bg="#f0f0f0")
        playback_frame.pack(pady=(2, 10), fill=tk.X)
        tk.Button(playback_frame, text="<< Play", command=self.play_backward, width=5).pack(side=tk.LEFT, expand=True, padx=1)
        tk.Button(playback_frame, text="Pause", command=self.pause, width=5, bg="#ffcccc").pack(side=tk.LEFT, expand=True, padx=1)
        tk.Button(playback_frame, text="Play >>", command=self.play_forward, width=5).pack(side=tk.LEFT, expand=True, padx=1)

        jump_frame = tk.Frame(control_frame, bg="#f0f0f0")
        jump_frame.pack(pady=5, fill=tk.X)
        tk.Label(jump_frame, text="Jump to:", bg="#f0f0f0").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(jump_frame, width=8)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        self.jump_entry.bind('<Return>', lambda event: self.jump_to_frame())
        tk.Button(jump_frame, text="Go", command=self.jump_to_frame).pack(side=tk.LEFT)

        self.frame_label = tk.Label(control_frame, text="Frame: 0 / 0", font=("Arial", 10, "bold"), bg="#f0f0f0")
        self.frame_label.pack(pady=10)

        instructions = (
            "CONTROLS:\n"
            "• Right-Click Drag -> Move Baseline\n"
            "• Left-Click Drag -> Draw New Box\n"
            "• Left-Click Inside -> Move Box\n"
            "• Left-Click Handles -> Resize Box\n"
            "• Press [ESC] -> Auto-detect / Delete\n\n"
            "Note: Interacting pauses playback."
        )
        tk.Label(control_frame, text=instructions, justify=tk.LEFT, fg="#555", bg="#f0f0f0").pack(pady=10, anchor=tk.W)

        tk.Button(control_frame, text="2. Export & Plot Single Case", command=self.trigger_export, bg="#2196F3", fg="white", font=("Arial", 10, "bold"), height=2).pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        self.canvas = tk.Canvas(self.parent, bg="black", width=800, height=600, cursor="cross")
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<B3-Motion>", self.on_right_drag) 

    def manual_prev(self): self.pause(); self.change_frame(-1)
    def manual_next(self): self.pause(); self.change_frame(1)
    def play_forward(self):
        self.play_direction = 1
        if not self.is_playing: self.is_playing = True; self.play_loop()
    def play_backward(self):
        self.play_direction = -1
        if not self.is_playing: self.is_playing = True; self.play_loop()
    def pause(self): self.is_playing = False

    def play_loop(self):
        if not self.is_playing: return
        if (self.play_direction == 1 and self.current_frame_idx >= self.num_frames - 1) or \
           (self.play_direction == -1 and self.current_frame_idx <= 0):
            self.pause()
            return
        self.change_frame(self.play_direction)
        self.root.after(self.playback_delay, self.play_loop)

    def clear_current_box(self, event=None):
        if self.current_frame_idx in self.bounding_boxes:
            del self.bounding_boxes[self.current_frame_idx]
            self.update_display()

    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Cine Files", "*.cine")])
        if not filepath: return
        try:
            metadata = cr.read_metadata(filepath)
            start_frame = metadata.FirstImageNo
            try: _, self.images, self.timestamps = cr.read(filepath, start=start_frame)
            except TypeError: _, self.images, self.timestamps = cr.read(filepath, start_frame, metadata.ImageCount)
            
            self.num_frames = len(self.images)
            self.fps = 1.0 / (self.timestamps[1] - self.timestamps[0]) if len(self.timestamps) > 1 else getattr(metadata, 'FrameRate1', 1000) 
            
            orig_h, orig_w = self.images[0].shape
            self.resize_factor = min(800 / orig_w, 600 / orig_h)
            if self.resize_factor > 1.0: self.resize_factor = 1.0 
                
            self.raw_baseline_y = int(orig_h - (50 / self.resize_factor))
            self.current_frame_idx = 0
            self.bounding_boxes = {}
            self.pause() 
            self.update_display()
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to load CINE file:\n{e}")

    def set_t0(self):
        self.t0_entry.delete(0, tk.END)
        self.t0_entry.insert(0, str(self.current_frame_idx))

    def change_frame(self, step):
        if self.images is not None:
            self.current_frame_idx = max(0, min(self.num_frames - 1, self.current_frame_idx + step))
            self.update_display()

    def jump_to_frame(self):
        if self.images is None: return
        self.pause() 
        try:
            target_frame = int(self.jump_entry.get())
            if 0 <= target_frame < self.num_frames:
                self.current_frame_idx = target_frame
                self.update_display()
        except ValueError: pass

    def update_display(self):
        if self.images is None: return
        raw_frame = self.images[self.current_frame_idx]
        
        if self.resize_factor < 1.0:
            new_w, new_h = int(raw_frame.shape[1] * self.resize_factor), int(raw_frame.shape[0] * self.resize_factor)
            display_frame = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else: display_frame = raw_frame

        frame_8bit = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2RGB)
        disp_baseline_y = int(self.raw_baseline_y * self.resize_factor)
        
        if self.current_frame_idx not in self.bounding_boxes:
            bbox = detect_droplet(frame_8bit, disp_baseline_y)
            if bbox:
                self.bounding_boxes[self.current_frame_idx] = (int(bbox[0]/self.resize_factor), int(bbox[1]/self.resize_factor), int(bbox[2]/self.resize_factor), int(bbox[3]/self.resize_factor))
        
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.create_line(0, disp_baseline_y, frame_rgb.shape[1], disp_baseline_y, fill="red", dash=(4, 4), width=2, tags="baseline")
        
        if self.current_frame_idx in self.bounding_boxes:
            rx, ry, rw, rh = self.bounding_boxes[self.current_frame_idx]
            dx, dy, dw, dh = int(rx*self.resize_factor), int(ry*self.resize_factor), int(rw*self.resize_factor), int(rh*self.resize_factor)
            self.canvas.create_rectangle(dx, dy, dx+dw, dy+dh, outline="#00FF00", width=2, tags="bbox")
            
            hs = 4 
            for hx, hy in [(dx, dy), (dx+dw//2, dy), (dx+dw, dy), (dx, dy+dh//2), (dx+dw, dy+dh//2), (dx, dy+dh), (dx+dw//2, dy+dh), (dx+dw, dy+dh)]:
                self.canvas.create_rectangle(hx-hs, hy-hs, hx+hs, hy+hs, fill="yellow", outline="black", tags="handle")
            
        self.frame_label.config(text=f"Frame: {self.current_frame_idx} / {self.num_frames - 1}")
        self.jump_entry.delete(0, tk.END)
        self.jump_entry.insert(0, str(self.current_frame_idx))

    def on_right_drag(self, event):
        if self.images is None: return
        self.pause() 
        max_disp_y = 600 if self.resize_factor == 1.0 else int(self.images[0].shape[0] * self.resize_factor)
        self.raw_baseline_y = int(max(0, min(event.y, max_disp_y)) / self.resize_factor)
        self.update_display()

    def on_left_press(self, event):
        if self.images is None: return
        self.pause() 
        self.start_x, self.start_y = event.x, event.y
        if self.current_frame_idx in self.bounding_boxes:
            rx, ry, rw, rh = self.bounding_boxes[self.current_frame_idx]
            dx, dy, dw, dh = rx*self.resize_factor, ry*self.resize_factor, rw*self.resize_factor, rh*self.resize_factor
            
            handles = {'TL': (dx, dy), 'T': (dx+dw//2, dy), 'TR': (dx+dw, dy), 'L': (dx, dy+dh//2), 'R': (dx+dw, dy+dh//2), 'BL': (dx, dy+dh), 'B': (dx+dw//2, dy+dh), 'BR': (dx+dw, dy+dh)}
            for name, (hx, hy) in handles.items():
                if abs(event.x - hx) <= 8 and abs(event.y - hy) <= 8:
                    self.interaction_mode, self.temp_box_cache = f'RESIZE_{name}', (rx, ry, rw, rh)
                    return
            if dx < event.x < dx+dw and dy < event.y < dy+dh:
                self.interaction_mode, self.temp_box_cache = 'MOVE', (rx, ry, rw, rh)
                return
        self.interaction_mode = 'DRAW'

    def on_left_drag(self, event):
        if not self.interaction_mode: return
        if self.interaction_mode == 'DRAW':
            self.canvas.delete("temp_bbox")
            self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="yellow", width=2, tags="temp_bbox")
        elif self.interaction_mode == 'MOVE':
            dx_raw, dy_raw = int((event.x - self.start_x)/self.resize_factor), int((event.y - self.start_y)/self.resize_factor)
            orig_x, orig_y, orig_w, orig_h = self.temp_box_cache
            self.bounding_boxes[self.current_frame_idx] = (orig_x + dx_raw, orig_y + dy_raw, orig_w, orig_h)
            self.update_display()
        elif self.interaction_mode.startswith('RESIZE_'):
            dx_raw, dy_raw = int((event.x - self.start_x)/self.resize_factor), int((event.y - self.start_y)/self.resize_factor)
            orig_x, orig_y, orig_w, orig_h = self.temp_box_cache
            new_x, new_y, new_w, new_h = orig_x, orig_y, orig_w, orig_h
            h_type = self.interaction_mode.split('_')[1]
            
            if 'L' in h_type: delta = min(dx_raw, orig_w-5); new_x = orig_x+delta; new_w = orig_w-delta
            if 'R' in h_type: new_w = max(5, orig_w+dx_raw)
            if 'T' in h_type: delta = min(dy_raw, orig_h-5); new_y = orig_y+delta; new_h = orig_h-delta
            if 'B' in h_type: new_h = max(5, orig_h+dy_raw)
            self.bounding_boxes[self.current_frame_idx] = (new_x, new_y, new_w, new_h)
            self.update_display()

    def on_left_release(self, event):
        if self.interaction_mode == 'DRAW':
            disp_w, disp_h = abs(event.x - self.start_x), abs(event.y - self.start_y)
            if disp_w > 5 and disp_h > 5:
                self.bounding_boxes[self.current_frame_idx] = (int(min(self.start_x, event.x)/self.resize_factor), int(min(self.start_y, event.y)/self.resize_factor), int(disp_w/self.resize_factor), int(disp_h/self.resize_factor))
                self.update_display()
        self.interaction_mode, self.temp_box_cache = None, None

    def trigger_export(self):
        try:
            px_per_mm, init_dia, t0_frame = float(self.scale_entry.get()), float(self.init_dia_entry.get()), int(self.t0_entry.get())
            if px_per_mm <= 0 or init_dia <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure Scale, Initial Dia, and Impact Frame are valid.")
            return
        self.pause() 
        export_and_plot_single(self.bounding_boxes, self.timestamps, self.fps, 1.0/px_per_mm, t0_frame, self.raw_baseline_y, init_dia)


# ==========================================
# MODULE 2: DATA COMPARATOR
# ==========================================
class ComparatorTab:
    def __init__(self, parent):
        self.parent = parent
        self.loaded_files = [] # Stores file paths
        
        self.setup_ui()

    def setup_ui(self):
        # Left Panel (Controls)
        control_frame = tk.Frame(self.parent, width=300, padx=15, pady=15, bg="#e8f4f8")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(control_frame, text="Comparison Module", font=("Arial", 14, "bold"), bg="#e8f4f8").pack(pady=(0, 15))

        tk.Button(control_frame, text="+ Load CSV Files", command=self.load_csvs, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), pady=5).pack(fill=tk.X)
        
        tk.Label(control_frame, text="Loaded Cases:", bg="#e8f4f8").pack(anchor=tk.W, pady=(15, 2))
        
        # Listbox for files
        self.listbox = tk.Listbox(control_frame, selectmode=tk.MULTIPLE, height=15)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        
        btn_frame = tk.Frame(control_frame, bg="#e8f4f8")
        btn_frame.pack(fill=tk.X, pady=5)
        tk.Button(btn_frame, text="Remove Selected", command=self.remove_selected).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(btn_frame, text="Clear All", command=self.clear_all).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=2)

        # Settings
        self.log_scale_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="Use Log Scale for Time (s)", variable=self.log_scale_var, bg="#e8f4f8").pack(anchor=tk.W, pady=15)

        tk.Button(control_frame, text="Generate Comparison Plot", command=self.plot, bg="#2196F3", fg="white", font=("Arial", 11, "bold"), height=2).pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        # Right Panel (Instructions)
        info_frame = tk.Frame(self.parent, padx=30, pady=30, bg="white")
        info_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        instructions = (
            "How to use the Multi-Case Comparator:\n\n"
            "1. Go to the 'Video Analysis' tab and process your videos.\n"
            "2. Save the resulting CSV files to a known folder.\n"
            "3. Come back to this tab and click '+ Load CSV Files'.\n"
            "4. Select as many CSV files as you want to compare.\n"
            "5. Click 'Generate Comparison Plot'.\n\n"
            "The system will automatically extract the non-dimensional height (h*) \n"
            "and spreading factor (d*) from your CSVs and plot them together \n"
            "using distinct colors and markers."
        )
        tk.Label(info_frame, text=instructions, font=("Arial", 12), justify=tk.LEFT, bg="white").pack(anchor=tk.NW)

    def load_csvs(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv")])
        for path in filepaths:
            if path not in self.loaded_files:
                self.loaded_files.append(path)
                filename = os.path.basename(path)
                self.listbox.insert(tk.END, filename)

    def remove_selected(self):
        selected_indices = self.listbox.curselection()
        for i in reversed(selected_indices):
            self.listbox.delete(i)
            del self.loaded_files[i]

    def clear_all(self):
        self.listbox.delete(0, tk.END)
        self.loaded_files.clear()

    def plot(self):
        if not self.loaded_files:
            messagebox.showwarning("No Data", "Please load at least one CSV file to compare.")
            return
        
        use_log = self.log_scale_var.get()
        plot_multi_comparison(self.loaded_files, use_log)

# ==========================================
# MAIN APPLICATION BOOTSTRAPPER
# ==========================================
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Droplet Impact Analyzer Suite")
        self.root.geometry("1150x650")

        # Create Tabbed Environment
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Build Frames for Tabs
        self.frame_analyzer = ttk.Frame(self.notebook)
        self.frame_comparator = ttk.Frame(self.notebook)

        self.notebook.add(self.frame_analyzer, text="1. Video Analysis")
        self.notebook.add(self.frame_comparator, text="2. Multi-Case Comparison")

        # Initialize Modules
        self.analyzer_module = VideoAnalyzerTab(self.frame_analyzer, self.root)
        self.comparator_module = ComparatorTab(self.frame_comparator)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()