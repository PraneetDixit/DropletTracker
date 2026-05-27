import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import os

# Use Matplotlib's Object-Oriented API for Tkinter embedding
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def export_and_plot(bounding_boxes, timestamps, fps, scale, t0_frame, baseline_y, init_dia):
    if not bounding_boxes:
        messagebox.showwarning("Empty Data", "No bounding box data to plot.")
        return

    # Prompt user for Save Location (Hooks into main.py's existing Tk root)
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        title="Save Droplet Kinematics Data As...",
        initialfile="droplet_kinematics.csv"
    )
    
    if not save_path:
        print("Export cancelled by user.")
        return

    results = []
    
    use_hardware_time = timestamps is not None and t0_frame < len(timestamps)
    if use_hardware_time:
        t0_time = timestamps[t0_frame]

    for frame_idx, (x, y, w, h) in sorted(bounding_boxes.items()):
        # Calculate accurate SECOND timing
        if use_hardware_time and frame_idx < len(timestamps):
            t_diff = timestamps[frame_idx] - t0_time
            if hasattr(t_diff, 'total_seconds'):
                time_s = t_diff.total_seconds()
            else:
                time_s = float(t_diff) / 1_000_000.0
        else:
            time_s = (frame_idx - t0_frame) / fps
            
        actual_height_px = baseline_y - y
        
        # Physical & Non-dimensional parameters
        h_mm = actual_height_px * scale
        dia_mm = w * scale
        h_star = h_mm / init_dia
        d_star = dia_mm / init_dia
        
        results.append([frame_idx, time_s, h_mm, dia_mm, h_star, d_star])

    # Save to CSV
    columns = ['Frame', 'Time_s', 'Height_mm', 'Diameter_mm', 'Height_nondim_h_star', 'Diameter_nondim_d_star']
    df = pd.DataFrame(results, columns=columns)
    
    try:
        df.to_csv(save_path, index=False)
        print(f"Successfully saved data to: {save_path}")
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save file:\n{e}")
        return

    # --- Split Data for Log Tabs ---
    # Pre-impact (t < 0) -> Convert to positive for log scale
    df_pre = df[df['Time_s'] < 0].copy()
    df_pre['Neg_Time_s'] = -df_pre['Time_s'] 
    
    # Post-impact (t > 0)
    df_post = df[df['Time_s'] > 0].copy()

    # --- BUILD THE UNIFIED DASHBOARD WINDOW ---
    filename_only = os.path.basename(save_path)
    
    dashboard = tk.Toplevel()
    dashboard.title(f"Results Dashboard - {filename_only}")
    dashboard.geometry("1100x700")
    
    # Create the Tab Manager
    notebook = ttk.Notebook(dashboard)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_plot_tab(parent_notebook, tab_title, data, x_col, x_label, is_log=False, invert_x=False):
        """Helper function to build a tab with embedded Matplotlib graphs"""
        frame = ttk.Frame(parent_notebook)
        parent_notebook.add(frame, text=tab_title)
        
        # Create a Matplotlib Figure (independent of pyplot state)
        fig = Figure(figsize=(12, 5), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Don't try to plot log scales if data is empty (e.g. no pre-impact frames)
        if data.empty:
            ax1.text(0.5, 0.5, "No data available for this timeframe", ha='center')
            ax2.text(0.5, 0.5, "No data available for this timeframe", ha='center')
        else:
            # Plot 1: Height
            ax1.scatter(data[x_col], data['Height_nondim_h_star'], color='dodgerblue', label='h* (h / D0)', s=20, alpha=0.8)
            ax1.set_ylabel('Non-dim Height ($h^*$)', fontsize=12)
            ax1.set_title('Height / Rebound Dynamics', fontsize=13)
            
            # Plot 2: Diameter
            ax2.scatter(data[x_col], data['Diameter_nondim_d_star'], color='crimson', label='d* (d / D0)', s=20, alpha=0.8)
            ax2.set_ylabel('Non-dim Spreading ($d^*$)', fontsize=12)
            ax2.set_title('Spreading Factor', fontsize=13)
            
            for ax in (ax1, ax2):
                ax.set_xlabel(x_label, fontsize=12)
                ax.grid(True, which="both", ls="--", alpha=0.4)
                ax.legend(loc='best')
                
                if is_log:
                    ax.set_xscale('log')
                else:
                    # Only draw the impact line on the linear scale
                    ax.axvline(x=0, color='black', linestyle='--', alpha=0.6, label='Impact (t=0)')
                    
                if invert_x:
                    ax.invert_xaxis()
        
        fig.tight_layout()
        
        # Embed the Figure into the Tkinter Frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        
        # Add the interactive toolbar (zoom, pan, save)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        
        # Pack them into the UI
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --- Generate the 3 Tabs ---
    create_plot_tab(
        notebook, 
        tab_title="1. Full Timeline (Linear)", 
        data=df, 
        x_col='Time_s', 
        x_label='Time (s)', 
        is_log=False
    )
    
    create_plot_tab(
        notebook, 
        tab_title="2. Pre-Impact Dynamics (Log)", 
        data=df_pre, 
        x_col='Neg_Time_s', 
        x_label='Time before impact (-t, seconds) [Log Scale]', 
        is_log=True, 
        invert_x=True # Time flows -> towards 0
    )
    
    create_plot_tab(
        notebook, 
        tab_title="3. Post-Impact Dynamics (Log)", 
        data=df_post, 
        x_col='Time_s', 
        x_label='Time after impact (t, seconds) [Log Scale]', 
        is_log=True
    )