import pandas as pd
import matplotlib.pyplot as plt
import tkinter.messagebox as messagebox

def export_and_plot(bounding_boxes, fps, scale, t0_frame, baseline_y):
    if not bounding_boxes:
        messagebox.showwarning("Empty Data", "No bounding box data to plot.")
        return

    results = []
    
    for frame_idx, (x, y, w, h) in sorted(bounding_boxes.items()):
        time_sec = (frame_idx - t0_frame) / fps
        actual_height_px = baseline_y - y
        h_mm = actual_height_px * scale
        dia_mm = w * scale
        results.append([frame_idx, time_sec, h_mm, dia_mm])

    df = pd.DataFrame(results, columns=['Frame', 'Time_s', 'Height_mm', 'Diameter_mm'])
    df.to_csv("droplet_kinematics.csv", index=False)
    print("Saved data to 'droplet_kinematics.csv'")

    plt.figure(figsize=(10, 6))
    plt.plot(df['Time_s'], df['Height_mm'], label='Height (mm)', color='dodgerblue', marker='.', linestyle='-')
    plt.plot(df['Time_s'], df['Diameter_mm'], label='Diameter (mm)', color='crimson', marker='.', linestyle='-')
    
    plt.axvline(x=0, color='black', linestyle='--', label='Impact (t=0)', alpha=0.6)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Measurements (mm)', fontsize=12)
    plt.title('Droplet Morphology: Pre and Post Impact', fontsize=14, pad=15)
    plt.legend(loc='best')
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.show()