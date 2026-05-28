import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import os
import itertools

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ==============================================================
# FUNCTION 1: SINGLE CASE EXPORT & DASHBOARD (Called from Tab 1)
# ==============================================================
def export_and_plot_single(bounding_boxes, timestamps, fps, scale, t0_frame, baseline_y, init_dia):
    if not bounding_boxes:
        messagebox.showwarning("Empty Data", "No bounding box data to plot.")
        return

    root = tk.Tk()
    root.withdraw()
    
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
        if use_hardware_time and frame_idx < len(timestamps):
            t_diff = timestamps[frame_idx] - t0_time
            if hasattr(t_diff, 'total_seconds'): time_s = t_diff.total_seconds()
            else: time_s = float(t_diff) / 1_000_000.0
        else:
            time_s = (frame_idx - t0_frame) / fps
            
        actual_height_px = baseline_y - y
        h_mm = actual_height_px * scale
        dia_mm = w * scale
        h_star = h_mm / init_dia
        d_star = dia_mm / init_dia
        
        results.append([frame_idx, time_s, h_mm, dia_mm, h_star, d_star])

    columns = ['Frame', 'Time_s', 'Height_mm', 'Diameter_mm', 'Height_nondim_h_star', 'Diameter_nondim_d_star']
    df = pd.DataFrame(results, columns=columns)
    
    try:
        df.to_csv(save_path, index=False)
        print(f"Successfully saved data to: {save_path}")
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save file:\n{e}")
        return

    df_pre = df[df['Time_s'] < 0].copy()
    df_pre['Neg_Time_s'] = -df_pre['Time_s'] 
    df_post = df[df['Time_s'] > 0].copy()

    filename_only = os.path.basename(save_path)
    dashboard = tk.Toplevel()
    dashboard.title(f"Results Dashboard - {filename_only}")
    dashboard.geometry("1100x700")
    
    notebook = ttk.Notebook(dashboard)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_plot_tab(parent_notebook, tab_title, data, x_col, x_label, is_log=False, invert_x=False):
        frame = ttk.Frame(parent_notebook)
        parent_notebook.add(frame, text=tab_title)
        
        fig = Figure(figsize=(12, 5), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        if data.empty:
            ax1.text(0.5, 0.5, "No data available", ha='center')
            ax2.text(0.5, 0.5, "No data available", ha='center')
        else:
            ax1.scatter(data[x_col], data['Height_nondim_h_star'], color='dodgerblue', label='h* (h / D0)', s=20, alpha=0.8)
            ax1.set_ylabel('Non-dim Height ($h^*$)', fontsize=12)
            ax1.set_title('Height / Rebound Dynamics', fontsize=13)
            
            ax2.scatter(data[x_col], data['Diameter_nondim_d_star'], color='crimson', label='d* (d / D0)', s=20, alpha=0.8)
            ax2.set_ylabel('Non-dim Spreading ($d^*$)', fontsize=12)
            ax2.set_title('Spreading Factor', fontsize=13)
            
            for ax in (ax1, ax2):
                ax.set_xlabel(x_label, fontsize=12)
                ax.grid(True, which="both", ls="--", alpha=0.4)
                ax.legend(loc='best')
                
                if is_log: ax.set_xscale('log')
                else: ax.axvline(x=0, color='black', linestyle='--', alpha=0.6, label='Impact (t=0)')
                if invert_x: ax.invert_xaxis()
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    create_plot_tab(notebook, "1. Full Timeline (Linear)", df, 'Time_s', 'Time (s)', False)
    create_plot_tab(notebook, "2. Pre-Impact (Log)", df_pre, 'Neg_Time_s', 'Time before impact (-t, s) [Log]', True, True)
    create_plot_tab(notebook, "3. Post-Impact (Log)", df_post, 'Time_s', 'Time after impact (t, s) [Log]', True)

# ==============================================================
# FUNCTION 2: MULTI-CASE COMPARISON (Called from Tab 2)
# ==============================================================
def plot_multi_comparison(csv_paths, use_log_scale):
    """Reads multiple CSVs and overlays them on a single Matplotlib figure using scatter plots"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    scale_text = "(Log Scale)" if use_log_scale else "(Linear Scale)"
    fig.canvas.manager.set_window_title("Multi-Case Comparison")
    fig.suptitle(f"Comparative Kinematics Overlay {scale_text}", fontsize=16, y=1.02)

    colors = plt.cm.tab10.colors  
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', '<', '>', 'p', '*'])
    
    failed_files = []
    plotted_count = 0

    for i, path in enumerate(csv_paths):
        name = os.path.basename(path).replace('.csv', '')
        try:
            df = pd.read_csv(path)
            
            if df.empty:
                failed_files.append((name, "CSV is completely empty."))
                continue
            
            # Strict format check based on new convention
            if 'Time_s' not in df.columns:
                failed_files.append((name, "Old CSV format. Requires 'Time_s' column."))
                continue
            
            if use_log_scale:
                df = df[df['Time_s'] > 0]
                if df.empty:
                    failed_files.append((name, "No post-impact data (t > 0) found for log scale."))
                    continue

            color = colors[plotted_count % len(colors)]
            marker = next(markers)
            
            # Changed from ax.plot(...) to ax.scatter(...)
            ax1.scatter(df['Time_s'], df['Height_nondim_h_star'], 
                        color=color, marker=marker, s=25, 
                        label=name, alpha=0.8)

            ax2.scatter(df['Time_s'], df['Diameter_nondim_d_star'], 
                        color=color, marker=marker, s=25, 
                        label=name, alpha=0.8)
                     
            plotted_count += 1

        except Exception as e:
            failed_files.append((name, f"File read error: {e}"))

    if failed_files:
        error_msg = "The following files were skipped:\n\n"
        for f_name, reason in failed_files:
            error_msg += f"• {f_name}: {reason}\n"
        messagebox.showwarning("Plotting Warnings", error_msg)
        
    if plotted_count == 0:
        plt.close(fig)
        return

    # Formatting Plot 1: Height
    ax1.set_ylabel('Non-dim Height ($h^*$)', fontsize=12)
    ax1.set_title('Height Evolution Overlay', fontsize=14)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(loc='best', fontsize=9)
    
    # Formatting Plot 2: Diameter
    ax2.set_ylabel('Non-dim Spreading Factor ($d^*$)', fontsize=12)
    ax2.set_title('Spreading Evolution Overlay', fontsize=14)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(loc='best', fontsize=9)

    for ax in (ax1, ax2):
        if use_log_scale:
            ax.set_xscale('log')
            ax.set_xlabel('Time (s) [Log Scale]', fontsize=12)
        else:
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.6) 

    plt.tight_layout()
    plt.show()