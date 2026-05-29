import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import os
import itertools

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def export_and_plot_single(bounding_boxes, timestamps, fps, scale, t0_frame, baseline_y, init_dia):
    if not bounding_boxes:
        messagebox.showwarning("Empty Data", "No bounding box data to plot.")
        return
    
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        title="Save Droplet Kinematics Data As...",
        initialfile="droplet_kinematics.csv"
    )
    
    if not save_path:
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
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save file:\n{e}")
        return

    df_pre = df[df['Time_s'] < 0].copy()
    df_pre['Neg_Time_s'] = -df_pre['Time_s'] 
    df_post = df[df['Time_s'] > 0].copy()

    filename_only = os.path.basename(save_path)
    
    # Use modern Toplevel
    dashboard = tb.Toplevel()
    dashboard.title(f"Results Dashboard - {filename_only}")
    dashboard.geometry("1100x700")
    
    notebook = tb.Notebook(dashboard, bootstyle=INFO)
    notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)

    def create_plot_tab(parent_notebook, tab_title, data, x_col, x_label, is_log=False, invert_x=False):
        frame = tb.Frame(parent_notebook)
        parent_notebook.add(frame, text=tab_title)
        
        # Use a dark background figure to match the theme
        plt.style.use('dark_background')
        fig = Figure(figsize=(12, 5), dpi=100)
        fig.patch.set_facecolor('#222222')
        
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        for ax in (ax1, ax2):
            ax.set_facecolor('#111111')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#555555')
        
        if data.empty:
            ax1.text(0.5, 0.5, "No data available", ha='center', color='white')
            ax2.text(0.5, 0.5, "No data available", ha='center', color='white')
        else:
            ax1.scatter(data[x_col], data['Height_nondim_h_star'], color='#00bcd4', label='h* (h / D0)', s=20, alpha=0.8)
            ax1.set_ylabel('Non-dim Height ($h^*$)', fontsize=12)
            ax1.set_title('Height / Rebound Dynamics', fontsize=13)
            
            ax2.scatter(data[x_col], data['Diameter_nondim_d_star'], color='#ff5252', label='d* (d / D0)', s=20, alpha=0.8)
            ax2.set_ylabel('Non-dim Spreading ($d^*$)', fontsize=12)
            ax2.set_title('Spreading Factor', fontsize=13)
            
            for ax in (ax1, ax2):
                ax.set_xlabel(x_label, fontsize=12)
                ax.grid(True, which="both", ls="--", alpha=0.2, color='white')
                ax.legend(loc='best', facecolor='#222222', edgecolor='#555555', labelcolor='white')
                
                if is_log: ax.set_xscale('log')
                else: ax.axvline(x=0, color='#888888', linestyle='--', alpha=0.8, label='Impact (t=0)')
                if invert_x: ax.invert_xaxis()
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    create_plot_tab(notebook, "  1. Full Timeline (Linear)  ", df, 'Time_s', 'Time (s)', False)
    create_plot_tab(notebook, "  2. Pre-Impact (Log)  ", df_pre, 'Neg_Time_s', 'Time before impact (-t, s) [Log]', True, True)
    create_plot_tab(notebook, "  3. Post-Impact (Log)  ", df_post, 'Time_s', 'Time after impact (t, s) [Log]', True)

# ==============================================================
# FUNCTION 2: MULTI-CASE COMPARISON (Called from Tab 2)
# ==============================================================
def plot_multi_comparison(csv_paths):
    valid_datasets = []
    failed_files = []
    
    for path in csv_paths:
        name = os.path.basename(path).replace('.csv', '')
        try:
            df = pd.read_csv(path)
            if df.empty:
                failed_files.append((name, "CSV is empty."))
                continue
            if 'Time_s' not in df.columns or 'Frame' not in df.columns:
                failed_files.append((name, "Missing 'Time_s' or 'Frame' format."))
                continue
                
            valid_datasets.append({'name': name, 'df_raw': df})
        except Exception as e:
            failed_files.append((name, f"Read error: {e}"))
            
    if failed_files:
        error_msg = "The following files were skipped:\n\n"
        for f_name, reason in failed_files:
            error_msg += f"• {f_name}: {reason}\n"
        messagebox.showwarning("Plotting Warnings", error_msg)
        
    if not valid_datasets:
        return
        
    global_min_frame = max([item['df_raw']['Frame'].min() for item in valid_datasets])
    global_max_frame = min([item['df_raw']['Frame'].max() for item in valid_datasets])

    colors = plt.cm.Set2.colors  # Brighter color map for dark themes
    markers = itertools.cycle(['o', 's', '^', 'D', 'v', '<', '>', 'p', '*'])
    
    for i, item in enumerate(valid_datasets):
        df = item['df_raw']
        df_sliced = df[(df['Frame'] >= global_min_frame) & (df['Frame'] <= global_max_frame)].copy()
        item['df_linear'] = df_sliced
        
        df_pre = df_sliced[df_sliced['Time_s'] < 0].copy()
        df_pre['Neg_Time_s'] = -df_pre['Time_s']
        item['df_pre'] = df_pre
        
        item['df_post'] = df_sliced[df_sliced['Time_s'] > 0].copy()
        
        item['color'] = colors[i % len(colors)]
        item['marker'] = next(markers)

    dashboard = tb.Toplevel()
    dashboard.title(f"Multi-Case Comparison Overlay ({len(valid_datasets)} Cases)")
    dashboard.geometry("1100x700")
    
    notebook = tb.Notebook(dashboard, bootstyle=INFO)
    notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
    
    def create_multi_plot_tab(parent_notebook, tab_title, data_key, x_col, x_label, is_log=False, invert_x=False):
        frame = tb.Frame(parent_notebook)
        parent_notebook.add(frame, text=tab_title)
        
        plt.style.use('dark_background')
        fig = Figure(figsize=(12, 5), dpi=100)
        fig.patch.set_facecolor('#222222')
        
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        for ax in (ax1, ax2):
            ax.set_facecolor('#111111')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#555555')
        
        has_data = False
        
        for item in valid_datasets:
            df_plot = item[data_key]
            if df_plot.empty: 
                continue
            has_data = True
            
            ax1.scatter(df_plot[x_col], df_plot['Height_nondim_h_star'], 
                        color=item['color'], marker=item['marker'], s=35, label=item['name'], alpha=0.9)
                        
            ax2.scatter(df_plot[x_col], df_plot['Diameter_nondim_d_star'], 
                        color=item['color'], marker=item['marker'], s=35, label=item['name'], alpha=0.9)
                        
        if not has_data:
            ax1.text(0.5, 0.5, "No data available in this timeframe", ha='center', color='white')
            ax2.text(0.5, 0.5, "No data available in this timeframe", ha='center', color='white')
        else:
            ax1.set_ylabel('Non-dim Height ($h^*$)', fontsize=12)
            ax1.set_title('Height Evolution Overlay', fontsize=13)
            ax2.set_ylabel('Non-dim Spreading ($d^*$)', fontsize=12)
            ax2.set_title('Spreading Evolution Overlay', fontsize=13)
            
            for ax in (ax1, ax2):
                ax.set_xlabel(x_label, fontsize=12)
                ax.grid(True, which="both", ls="--", alpha=0.2, color='white')
                ax.legend(loc='best', fontsize=9, facecolor='#222222', edgecolor='#555555', labelcolor='white')
                
                if is_log: ax.set_xscale('log')
                else: ax.axvline(x=0, color='#888888', linestyle='--', alpha=0.8)
                if invert_x: ax.invert_xaxis()
                
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)

    create_multi_plot_tab(notebook, "  1. Linear Overlay  ", 'df_linear', 'Time_s', 'Time (s)', False)
    create_multi_plot_tab(notebook, "  2. Pre-Impact Overlay (Log)  ", 'df_pre', 'Neg_Time_s', 'Time before impact (-t, s) [Log]', True, True)
    create_multi_plot_tab(notebook, "  3. Post-Impact Overlay (Log)  ", 'df_post', 'Time_s', 'Time after impact (t, s) [Log]', True)