import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import cv2
import numpy as np
from PIL import Image, ImageTk
import cinereader as cr
import os
import math

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
        self.current_delay = 50 
        
        self.setup_ui()
        self.root.bind("<Escape>", self.clear_current_box)

    def setup_ui(self):
        # ---------------------------------------------------------
        # MODERN SIDEBAR (Left Panel)
        # ---------------------------------------------------------
        sidebar = tb.Frame(self.parent, width=320, padding=20)
        sidebar.pack(side=LEFT, fill=Y)
        sidebar.pack_propagate(False)

        tb.Label(sidebar, text="⚙️ Configuration", font=("Helvetica", 14, "bold")).pack(anchor=W, pady=(0, 10))
        tb.Separator(sidebar).pack(fill=X, pady=(0, 15))

        tb.Button(sidebar, text="📂 Load .CINE Video", command=self.load_file, bootstyle="success", width=25).pack(fill=X, pady=(0, 20), ipady=5)

        # File Properties Grid
        prop_frame = tb.Frame(sidebar)
        prop_frame.pack(fill=X, pady=(0, 20))
        
        tb.Label(prop_frame, text="Scale (1mm = px):", font=("Helvetica", 10)).grid(row=0, column=0, sticky=W, pady=5)
        self.scale_entry = tb.Entry(prop_frame, width=10, justify="center")
        self.scale_entry.insert(0, "50")
        self.scale_entry.grid(row=0, column=1, sticky=E, pady=5)

        tb.Label(prop_frame, text="Initial Dia (mm):", font=("Helvetica", 10)).grid(row=1, column=0, sticky=W, pady=5)
        self.init_dia_entry = tb.Entry(prop_frame, width=10, justify="center")
        self.init_dia_entry.insert(0, "2.52")
        self.init_dia_entry.grid(row=1, column=1, sticky=E, pady=5)

        # Vision Settings
        tb.Label(sidebar, text="👁️ Computer Vision", font=("Helvetica", 14, "bold")).pack(anchor=W, pady=(10, 10))
        tb.Separator(sidebar).pack(fill=X, pady=(0, 15))

        # --- UPDATED: Precision Controls with Cache Clearing ---
        tb.Label(sidebar, text="Detection Threshold").pack(anchor=W)
        f_thresh = tb.Frame(sidebar)
        f_thresh.pack(fill=X, pady=(5, 15))
        tb.Button(f_thresh, text="◀", command=lambda: self.adjust_thresh(-1), bootstyle="secondary", padding=(4,2)).pack(side=LEFT)
        self.thresh_slider = tb.Scale(f_thresh, from_=0, to=255, orient=HORIZONTAL, bootstyle=PRIMARY, command=self.on_cv_change)
        self.thresh_slider.set(100)
        self.thresh_slider.pack(side=LEFT, fill=X, expand=True, padx=5)
        tb.Button(f_thresh, text="▶", command=lambda: self.adjust_thresh(1), bootstyle="secondary", padding=(4,2)).pack(side=RIGHT)

        f_tilt_lbl = tb.Frame(sidebar)
        f_tilt_lbl.pack(fill=X)
        tb.Label(f_tilt_lbl, text="Substrate Tilt (°)").pack(side=LEFT)
        tb.Button(f_tilt_lbl, text="📐 Draw", command=self.enable_tilt_measure, bootstyle="warning-outline", padding=(5,2)).pack(side=RIGHT)
        
        f_tilt = tb.Frame(sidebar)
        f_tilt.pack(fill=X, pady=(5, 15))
        tb.Button(f_tilt, text="◀", command=lambda: self.adjust_tilt(-0.1), bootstyle="secondary", padding=(4,2)).pack(side=LEFT)
        self.tilt_slider = tb.Scale(f_tilt, from_=-45.0, to=45.0, orient=HORIZONTAL, bootstyle=PRIMARY, command=self.on_cv_change)
        self.tilt_slider.set(0.0)
        self.tilt_slider.pack(side=LEFT, fill=X, expand=True, padx=5)
        tb.Button(f_tilt, text="▶", command=lambda: self.adjust_tilt(0.1), bootstyle="secondary", padding=(4,2)).pack(side=RIGHT)

        f_base = tb.Frame(sidebar)
        f_base.pack(fill=X, pady=(0, 20))
        tb.Label(f_base, text="Baseline Shift:").pack(side=LEFT)
        tb.Button(f_base, text="▼", command=lambda: self.adjust_baseline(1), bootstyle="secondary").pack(side=RIGHT)
        tb.Button(f_base, text="▲", command=lambda: self.adjust_baseline(-1), bootstyle="secondary").pack(side=RIGHT, padx=5)
        # -------------------------------------------------------

        # Export Button
        tb.Button(sidebar, text="📊 Export & Plot Case", command=self.trigger_export, bootstyle="primary").pack(fill=X, side=BOTTOM, ipady=8)
        
        help_text = "R-Click: Move Base\nL-Click: Draw/Edit Box\nESC: Delete Box"
        tb.Label(sidebar, text=help_text, font=("Helvetica", 8), foreground="#888", justify=LEFT).pack(side=BOTTOM, pady=20, anchor=W)

        # ---------------------------------------------------------
        # MAIN CONTENT AREA (Right Panel)
        # ---------------------------------------------------------
        main_area = tb.Frame(self.parent, padding=20)
        main_area.pack(side=RIGHT, fill=BOTH, expand=True)

        # Top Info Bar
        top_bar = tb.Frame(main_area)
        top_bar.pack(fill=X, pady=(0, 10))
        
        self.frame_label = tb.Label(top_bar, text="FRAME: 0 / 0", font=("Helvetica", 16, "bold"), bootstyle="inverse-primary", padding=10)
        self.frame_label.pack(side=LEFT)

        f_t0 = tb.Frame(top_bar)
        f_t0.pack(side=RIGHT)
        tb.Label(f_t0, text="Impact Anchor (t=0):", font=("Helvetica", 10, "bold")).pack(side=LEFT, padx=10)
        self.t0_entry = tb.Entry(f_t0, width=6, justify="center")
        self.t0_entry.insert(0, "0")
        self.t0_entry.pack(side=LEFT)
        tb.Button(f_t0, text="Set Current", command=self.set_t0, bootstyle="info-outline").pack(side=LEFT, padx=10)

        # The Video Player Canvas
        canvas_container = tb.Frame(main_area, bootstyle="dark")
        canvas_container.pack(fill=BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_container, bg="#111111", width=800, height=500, cursor="cross", highlightthickness=0, borderwidth=0)
        self.canvas.pack(fill=BOTH, expand=True, padx=2, pady=2) 
        
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<B3-Motion>", self.on_right_drag) 

        # Media Player Controls
        player_bar = tb.Frame(main_area, padding=(0, 15, 0, 0))
        player_bar.pack(fill=X)

        f_jump = tb.Frame(player_bar)
        f_jump.pack(side=LEFT)
        tb.Label(f_jump, text="Jump:").pack(side=LEFT)
        self.jump_entry = tb.Entry(f_jump, width=6, justify="center")
        self.jump_entry.pack(side=LEFT, padx=5)
        self.jump_entry.bind('<Return>', lambda event: self.jump_to_frame())
        tb.Button(f_jump, text="Go", command=self.jump_to_frame, bootstyle="secondary").pack(side=LEFT)

        f_play = tb.Frame(player_bar)
        f_play.pack(side=RIGHT) 
        
        tb.Button(f_play, text="-1", command=self.manual_prev, bootstyle="secondary-outline", width=3).pack(side=LEFT, padx=2)
        tb.Button(f_play, text="⏮", command=lambda: self.start_playback(-1, 25), bootstyle="secondary", width=3).pack(side=LEFT, padx=2)
        tb.Button(f_play, text="◀", command=lambda: self.start_playback(-1, 50), bootstyle="secondary", width=3).pack(side=LEFT, padx=2)
        tb.Button(f_play, text="⏸", command=self.pause, bootstyle="danger", width=4).pack(side=LEFT, padx=2)
        tb.Button(f_play, text="▶", command=lambda: self.start_playback(1, 50), bootstyle="secondary", width=3).pack(side=LEFT, padx=2)
        tb.Button(f_play, text="⏭", command=lambda: self.start_playback(1, 25), bootstyle="secondary", width=3).pack(side=LEFT, padx=2)
        tb.Button(f_play, text="+1", command=self.manual_next, bootstyle="secondary-outline", width=3).pack(side=LEFT, padx=2)

    # --- LOGIC ---
    def adjust_thresh(self, delta):
        new_val = max(0, min(255, self.thresh_slider.get() + delta))
        self.thresh_slider.set(new_val)

    def adjust_tilt(self, delta):
        new_val = max(-45.0, min(45.0, self.tilt_slider.get() + delta))
        self.tilt_slider.set(new_val)

    def on_cv_change(self, *args):
        # Triggers whenever threshold or tilt changes. 
        # Clears cache so playback auto-detects dynamically.
        self.pause()
        self.bounding_boxes.clear() 
        self.update_display()

    def adjust_baseline(self, delta):
        if self.images is None: return
        self.pause()
        self.raw_baseline_y += delta
        max_y = self.images[0].shape[0]
        self.raw_baseline_y = max(0, min(self.raw_baseline_y, max_y))
        self.bounding_boxes.clear() # Clear cache here too
        self.update_display()

    def enable_tilt_measure(self):
        self.pause()
        self.interaction_mode = 'MEASURE_TILT'

    def start_playback(self, direction, delay):
        self.play_direction = direction
        self.current_delay = delay
        if not self.is_playing:
            self.is_playing = True
            self.play_loop()

    def manual_prev(self): self.pause(); self.change_frame(-1)
    def manual_next(self): self.pause(); self.change_frame(1)
    def pause(self): self.is_playing = False

    def play_loop(self):
        if not self.is_playing: return
        if (self.play_direction == 1 and self.current_frame_idx >= self.num_frames - 1) or \
           (self.play_direction == -1 and self.current_frame_idx <= 0):
            self.pause()
            return
        self.change_frame(self.play_direction)
        self.root.after(self.current_delay, self.play_loop)

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
            self.tilt_slider.set(0.0) 
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
        
        angle = self.tilt_slider.get()
            
        if angle != 0.0:
            h, w = raw_frame.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            raw_frame = cv2.warpAffine(raw_frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        if self.resize_factor < 1.0:
            new_w, new_h = int(raw_frame.shape[1] * self.resize_factor), int(raw_frame.shape[0] * self.resize_factor)
            display_frame = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else: display_frame = raw_frame

        frame_8bit = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2RGB)
        disp_baseline_y = int(self.raw_baseline_y * self.resize_factor)
        
        if self.current_frame_idx not in self.bounding_boxes:
            thresh_val = self.thresh_slider.get()
            bbox = detect_droplet(frame_8bit, disp_baseline_y, thresh_val)
            if bbox:
                self.bounding_boxes[self.current_frame_idx] = (int(bbox[0]/self.resize_factor), int(bbox[1]/self.resize_factor), int(bbox[2]/self.resize_factor), int(bbox[3]/self.resize_factor))
        
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.canvas.create_line(0, disp_baseline_y, frame_rgb.shape[1], disp_baseline_y, fill="#ff4444", dash=(4, 4), width=2, tags="baseline")
        
        if self.current_frame_idx in self.bounding_boxes:
            rx, ry, rw, rh = self.bounding_boxes[self.current_frame_idx]
            dx, dy, dw, dh = int(rx*self.resize_factor), int(ry*self.resize_factor), int(rw*self.resize_factor), int(rh*self.resize_factor)
            self.canvas.create_rectangle(dx, dy, dx+dw, dy+dh, outline="#00e676", width=2, tags="bbox")
            
            hs = 4 
            for hx, hy in [(dx, dy), (dx+dw//2, dy), (dx+dw, dy), (dx, dy+dh//2), (dx+dw, dy+dh//2), (dx, dy+dh), (dx+dw//2, dy+dh), (dx+dw, dy+dh)]:
                self.canvas.create_rectangle(hx-hs, hy-hs, hx+hs, hy+hs, fill="#ffeb3b", outline="black", tags="handle")
            
        self.frame_label.config(text=f"FRAME: {self.current_frame_idx} / {self.num_frames - 1}")
        self.jump_entry.delete(0, tk.END)
        self.jump_entry.insert(0, str(self.current_frame_idx))

    def on_right_drag(self, event):
        if self.images is None: return
        self.pause() 
        max_disp_y = 600 if self.resize_factor == 1.0 else int(self.images[0].shape[0] * self.resize_factor)
        self.raw_baseline_y = int(max(0, min(event.y, max_disp_y)) / self.resize_factor)
        self.bounding_boxes.clear() # Clear cache on drag
        self.update_display()

    def on_left_press(self, event):
        if self.images is None: return
        self.pause() 
        self.start_x, self.start_y = event.x, event.y
        
        if self.interaction_mode == 'MEASURE_TILT':
            return

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

        if self.interaction_mode == 'MEASURE_TILT':
            self.canvas.delete("temp_tilt")
            self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill="#00bcd4", width=2, tags="temp_tilt")
            return

        if self.interaction_mode == 'DRAW':
            self.canvas.delete("temp_bbox")
            self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="#ffeb3b", width=2, tags="temp_bbox")
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
        if self.interaction_mode == 'MEASURE_TILT':
            dy = event.y - self.start_y
            dx = event.x - self.start_x
            if dx != 0:
                if dx < 0: dx, dy = -dx, -dy  
                angle = math.degrees(math.atan2(dy, dx))
                self.tilt_slider.set(round(angle, 2))
            self.interaction_mode = None
            return

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
        self.loaded_files = [] 
        
        self.setup_ui()

    def setup_ui(self):
        sidebar = tb.Frame(self.parent, width=320, padding=20)
        sidebar.pack(side=LEFT, fill=Y)
        sidebar.pack_propagate(False)

        tb.Label(sidebar, text="📈 Comparison Engine", font=("Helvetica", 14, "bold")).pack(anchor=W, pady=(0, 10))
        tb.Separator(sidebar).pack(fill=X, pady=(0, 15))

        tb.Button(sidebar, text="➕ Add CSV Files", command=self.load_csvs, bootstyle="success", width=25).pack(fill=X, pady=(0, 20), ipady=5)
        
        tb.Label(sidebar, text="Loaded Cases:", font=("Helvetica", 10, "bold")).pack(anchor=W, pady=(0, 5))
        
        self.listbox = tk.Listbox(sidebar, selectmode=tk.MULTIPLE, bg="#1a1a1a", fg="white", borderwidth=0, highlightthickness=1, highlightcolor="#333", highlightbackground="#333", font=("Helvetica", 9))
        self.listbox.pack(fill=BOTH, expand=True)
        
        btn_frame = tb.Frame(sidebar)
        btn_frame.pack(fill=X, pady=10)
        tb.Button(btn_frame, text="Remove", command=self.remove_selected, bootstyle="warning-outline").pack(side=LEFT, expand=True, fill=X, padx=(0, 5))
        tb.Button(btn_frame, text="Clear All", command=self.clear_all, bootstyle="danger-outline").pack(side=RIGHT, expand=True, fill=X, padx=(5, 0))

        tb.Button(sidebar, text="🚀 Generate Dash", command=self.plot, bootstyle="primary").pack(fill=X, side=BOTTOM, ipady=8)

        main_area = tb.Frame(self.parent, padding=40)
        main_area.pack(side=RIGHT, fill=BOTH, expand=True)
        
        welcome_frame = tb.Frame(main_area)
        welcome_frame.place(relx=0.5, rely=0.5, anchor=CENTER)

        tb.Label(welcome_frame, text="Data Comparator", font=("Helvetica", 28, "bold"), foreground="#444").pack(pady=(0, 10))
        instructions = (
            "1. Process individual videos in the 'Video Analysis' tab.\n"
            "2. Add the exported .csv files to the sidebar on the left.\n"
            "3. Click 'Generate Dash' to render synchronized overlays."
        )
        tb.Label(welcome_frame, text=instructions, font=("Helvetica", 12), foreground="#777", justify=CENTER).pack()

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
        plot_multi_comparison(self.loaded_files)

# ==========================================
# MAIN APPLICATION BOOTSTRAPPER
# ==========================================
class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Droplet Impact Analyzer Suite")
        self.root.geometry("1200x750")

        self.notebook = tb.Notebook(root, bootstyle=INFO)
        self.notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

        self.frame_analyzer = tb.Frame(self.notebook)
        self.frame_comparator = tb.Frame(self.notebook)

        self.notebook.add(self.frame_analyzer, text="  1. Video Analysis  ")
        self.notebook.add(self.frame_comparator, text="  2. Multi-Case Comparison  ")

        self.analyzer_module = VideoAnalyzerTab(self.frame_analyzer, self.root)
        self.comparator_module = ComparatorTab(self.frame_comparator)

if __name__ == "__main__":
    root = tb.Window(themename="darkly") 
    app = MainApp(root)
    root.mainloop()