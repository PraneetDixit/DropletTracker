import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import cinereader as cr

from image_processor import detect_droplet
from data_exporter import export_and_plot

class DropletTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Droplet Impact Analyzer")
        
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

        # --- Playback State Variables ---
        self.is_playing = False
        self.play_direction = 1
        self.playback_delay = 50  # ~20 FPS (50 milliseconds per frame)
        
        self.setup_ui()
        self.root.bind("<Escape>", self.clear_current_box)

    def setup_ui(self):
        control_frame = tk.Frame(self.root, width=250, padx=15, pady=15, bg="#f0f0f0")
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

        # --- Manual Navigation ---
        nav_frame = tk.Frame(control_frame, bg="#f0f0f0")
        nav_frame.pack(pady=(15, 2), fill=tk.X)
        tk.Button(nav_frame, text="< Prev", command=self.manual_prev).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(nav_frame, text="Next >", command=self.manual_next).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=2)

        # --- NEW: Playback Engine Controls ---
        playback_frame = tk.Frame(control_frame, bg="#f0f0f0")
        playback_frame.pack(pady=(2, 10), fill=tk.X)
        tk.Button(playback_frame, text="<< Play", command=self.play_backward, width=5).pack(side=tk.LEFT, expand=True, padx=1)
        tk.Button(playback_frame, text="Pause", command=self.pause, width=5, bg="#ffcccc").pack(side=tk.LEFT, expand=True, padx=1)
        tk.Button(playback_frame, text="Play >>", command=self.play_forward, width=5).pack(side=tk.LEFT, expand=True, padx=1)
        # -------------------------------------

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

        tk.Button(control_frame, text="2. Export & Plot", command=self.trigger_export, bg="#2196F3", fg="white", font=("Arial", 10, "bold"), height=2).pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        self.canvas = tk.Canvas(self.root, bg="black", width=800, height=600, cursor="cross")
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<B3-Motion>", self.on_right_drag) 

    # --- PLAYBACK ENGINE ---
    def manual_prev(self):
        self.pause()
        self.change_frame(-1)

    def manual_next(self):
        self.pause()
        self.change_frame(1)

    def play_forward(self):
        self.play_direction = 1
        if not self.is_playing:
            self.is_playing = True
            self.play_loop()

    def play_backward(self):
        self.play_direction = -1
        if not self.is_playing:
            self.is_playing = True
            self.play_loop()

    def pause(self):
        self.is_playing = False

    def play_loop(self):
        if not self.is_playing:
            return

        # Check bounds: Stop playing if we hit the end or beginning
        if (self.play_direction == 1 and self.current_frame_idx >= self.num_frames - 1) or \
           (self.play_direction == -1 and self.current_frame_idx <= 0):
            self.pause()
            return

        # Advance frame
        self.change_frame(self.play_direction)
        
        # Schedule the next frame update 
        # (This keeps the GUI responsive while video plays)
        self.root.after(self.playback_delay, self.play_loop)
    # -----------------------

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
            frame_count = metadata.ImageCount
            
            try:
                _, self.images, self.timestamps = cr.read(filepath, start=start_frame)
            except TypeError:
                _, self.images, self.timestamps = cr.read(filepath, start_frame, frame_count)
            
            self.num_frames = len(self.images)
            self.fps = 1.0 / (self.timestamps[1] - self.timestamps[0]) if len(self.timestamps) > 1 else getattr(metadata, 'FrameRate1', 1000) 
            
            orig_h, orig_w = self.images[0].shape
            max_w, max_h = 800, 600
            self.resize_factor = min(max_w / orig_w, max_h / orig_h)
            if self.resize_factor > 1.0: self.resize_factor = 1.0 
                
            self.raw_baseline_y = int(orig_h - (50 / self.resize_factor))
            self.current_frame_idx = 0
            self.bounding_boxes = {}
            self.pause() # Reset playback state on load
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
        self.pause() # Auto-pause when jumping
        try:
            target_frame = int(self.jump_entry.get())
            if 0 <= target_frame < self.num_frames:
                self.current_frame_idx = target_frame
                self.update_display()
            else:
                messagebox.showwarning("Out of Bounds", f"Enter a frame between 0 and {self.num_frames - 1}.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid frame number.")

    def update_display(self):
        if self.images is None: return
        
        raw_frame = self.images[self.current_frame_idx]
        
        if self.resize_factor < 1.0:
            new_w = int(raw_frame.shape[1] * self.resize_factor)
            new_h = int(raw_frame.shape[0] * self.resize_factor)
            display_frame = cv2.resize(raw_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            display_frame = raw_frame

        frame_8bit = cv2.normalize(display_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        frame_rgb = cv2.cvtColor(frame_8bit, cv2.COLOR_GRAY2RGB)
        
        disp_baseline_y = int(self.raw_baseline_y * self.resize_factor)
        
        if self.current_frame_idx not in self.bounding_boxes:
            bbox = detect_droplet(frame_8bit, disp_baseline_y)
            if bbox:
                raw_x = int(bbox[0] / self.resize_factor)
                raw_y = int(bbox[1] / self.resize_factor)
                raw_w = int(bbox[2] / self.resize_factor)
                raw_h = int(bbox[3] / self.resize_factor)
                self.bounding_boxes[self.current_frame_idx] = (raw_x, raw_y, raw_w, raw_h)
        
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        width = frame_rgb.shape[1]
        self.canvas.create_line(0, disp_baseline_y, width, disp_baseline_y, fill="red", dash=(4, 4), width=2, tags="baseline")
        
        if self.current_frame_idx in self.bounding_boxes:
            raw_x, raw_y, raw_w, raw_h = self.bounding_boxes[self.current_frame_idx]
            disp_x = int(raw_x * self.resize_factor)
            disp_y = int(raw_y * self.resize_factor)
            disp_w = int(raw_w * self.resize_factor)
            disp_h = int(raw_h * self.resize_factor)
            
            self.canvas.create_rectangle(disp_x, disp_y, disp_x+disp_w, disp_y+disp_h, outline="#00FF00", width=2, tags="bbox")
            
            hs = 4 
            handle_coords = [
                (disp_x, disp_y),                     
                (disp_x + disp_w//2, disp_y),         
                (disp_x + disp_w, disp_y),            
                (disp_x, disp_y + disp_h//2),         
                (disp_x + disp_w, disp_y + disp_h//2),
                (disp_x, disp_y + disp_h),            
                (disp_x + disp_w//2, disp_y + disp_h),
                (disp_x + disp_w, disp_y + disp_h)    
            ]
            for hx, hy in handle_coords:
                self.canvas.create_rectangle(hx - hs, hy - hs, hx + hs, hy + hs, fill="yellow", outline="black", tags="handle")
            
        self.frame_label.config(text=f"Frame: {self.current_frame_idx} / {self.num_frames - 1}")
        
        self.jump_entry.delete(0, tk.END)
        self.jump_entry.insert(0, str(self.current_frame_idx))

    def on_right_drag(self, event):
        if self.images is None: return
        self.pause() # Auto-pause when adjusting baseline
        
        max_disp_y = 600 if self.resize_factor == 1.0 else int(self.images[0].shape[0] * self.resize_factor)
        safe_y = max(0, min(event.y, max_disp_y))
        
        self.raw_baseline_y = int(safe_y / self.resize_factor)
        self.update_display()

    def on_left_press(self, event):
        if self.images is None: return
        self.pause() # Auto-pause when user interacts with the canvas
        
        self.start_x = event.x
        self.start_y = event.y
        
        if self.current_frame_idx in self.bounding_boxes:
            rx, ry, rw, rh = self.bounding_boxes[self.current_frame_idx]
            dx = rx * self.resize_factor
            dy = ry * self.resize_factor
            dw = rw * self.resize_factor
            dh = rh * self.resize_factor
            
            hit_tolerance = 8
            handles = {
                'TL': (dx, dy),
                'T':  (dx + dw//2, dy),
                'TR': (dx + dw, dy),
                'L':  (dx, dy + dh//2),
                'R':  (dx + dw, dy + dh//2),
                'BL': (dx, dy + dh),
                'B':  (dx + dw//2, dy + dh),
                'BR': (dx + dw, dy + dh)
            }
            
            for name, (hx, hy) in handles.items():
                if abs(event.x - hx) <= hit_tolerance and abs(event.y - hy) <= hit_tolerance:
                    self.interaction_mode = f'RESIZE_{name}'
                    self.temp_box_cache = (rx, ry, rw, rh)
                    return
            
            if dx < event.x < dx + dw and dy < event.y < dy + dh:
                self.interaction_mode = 'MOVE'
                self.temp_box_cache = (rx, ry, rw, rh)
                return

        self.interaction_mode = 'DRAW'

    def on_left_drag(self, event):
        if not self.interaction_mode: return
            
        if self.interaction_mode == 'DRAW':
            self.canvas.delete("temp_bbox")
            self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="yellow", width=2, tags="temp_bbox")
            
        elif self.interaction_mode == 'MOVE':
            dx_raw = int((event.x - self.start_x) / self.resize_factor)
            dy_raw = int((event.y - self.start_y) / self.resize_factor)
            orig_x, orig_y, orig_w, orig_h = self.temp_box_cache
            self.bounding_boxes[self.current_frame_idx] = (orig_x + dx_raw, orig_y + dy_raw, orig_w, orig_h)
            self.update_display()
            
        elif self.interaction_mode.startswith('RESIZE_'):
            dx_raw = int((event.x - self.start_x) / self.resize_factor)
            dy_raw = int((event.y - self.start_y) / self.resize_factor)
            
            orig_x, orig_y, orig_w, orig_h = self.temp_box_cache
            new_x, new_y, new_w, new_h = orig_x, orig_y, orig_w, orig_h
            
            handle_type = self.interaction_mode.split('_')[1]
            
            if 'L' in handle_type:
                delta = min(dx_raw, orig_w - 5) 
                new_x = orig_x + delta
                new_w = orig_w - delta
            if 'R' in handle_type:
                new_w = max(5, orig_w + dx_raw)
            if 'T' in handle_type:
                delta = min(dy_raw, orig_h - 5)
                new_y = orig_y + delta
                new_h = orig_h - delta
            if 'B' in handle_type:
                new_h = max(5, orig_h + dy_raw)
                
            self.bounding_boxes[self.current_frame_idx] = (new_x, new_y, new_w, new_h)
            self.update_display()

    def on_left_release(self, event):
        if self.interaction_mode == 'DRAW':
            disp_x = min(self.start_x, event.x)
            disp_y = min(self.start_y, event.y)
            disp_w = abs(event.x - self.start_x)
            disp_h = abs(event.y - self.start_y)
            
            if disp_w > 5 and disp_h > 5:
                raw_x = int(disp_x / self.resize_factor)
                raw_y = int(disp_y / self.resize_factor)
                raw_w = int(disp_w / self.resize_factor)
                raw_h = int(disp_h / self.resize_factor)
                self.bounding_boxes[self.current_frame_idx] = (raw_x, raw_y, raw_w, raw_h)
                self.update_display()
                
        self.interaction_mode = None
        self.temp_box_cache = None

    def trigger_export(self):
        try:
            px_per_mm = float(self.scale_entry.get())
            init_dia = float(self.init_dia_entry.get())
            t0_frame = int(self.t0_entry.get())

            if px_per_mm <= 0 or init_dia <= 0:
                raise ValueError
            
            scale = 1.0 / px_per_mm
            
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure Scale (>0), Initial Dia (>0), and Impact Frame are valid numbers.")
            return
            
        self.pause() # Pause playback if hitting export
        export_and_plot(self.bounding_boxes, self.timestamps, self.fps, scale, t0_frame, self.raw_baseline_y, init_dia)

if __name__ == "__main__":
    root = tk.Tk()
    app = DropletTrackerApp(root)
    root.mainloop()