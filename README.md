# Droplet Impact Analyzer Suite

A comprehensive, Python-based desktop application for tracking, analyzing, and comparing high-speed droplet impact dynamics. Designed specifically for `.cine` files (Phantom High-Speed Cameras), this suite provides automated edge detection, interactive bounding box correction, and automated non-dimensional kinematics plotting.

## Features

* **Native `.cine` File Support:** Directly reads raw high-speed camera files, preserving precision hardware timestamps (microseconds) to prevent frame-drift errors.
* **Computer Vision Auto-Tracking:** Uses OpenCV adaptive thresholding and morphological filtering to automatically detect the droplet above a user-defined substrate baseline.
* **Interactive UI:** 8-point resize handles and drag-and-drop bounding boxes allow for pixel-perfect manual corrections when auto-detection fails (e.g., due to contact line reflections).
* **Playback Engine:** Includes video scrubbing, play/pause, step forward/backward, and a specific "Jump to Frame" tool for navigating files with thousands of frames.
* **Physical & Non-Dimensional Calibration:** Converts pixel data to physical dimensions (mm) and automatically calculates non-dimensional spreading factor ($d^*$) and height ($h^*$) based on initial diameter ($D_0$).
* **Results Dashboard:** Embedded Matplotlib graphs with a tabbed UI. Generates linear full-timeline plots, and inverted log-scale plots for pre- and post-impact microsecond dynamics.
* **Multi-Case Comparator:** Load multiple exported `.csv` files into a unified overlay graph to instantly compare the spreading and rebound dynamics of different droplet impact cases.

## File Structure

* `main.py`: The core Tkinter GUI, playback engine, event loops, and interaction state machine.
* `image_processor.py`: Contains the OpenCV logic (`detect_droplet`) for isolating the droplet from the background and filtering noise.
* `data_exporter.py`: Handles all time conversions, physical scaling math, CSV generation, and the embedded Matplotlib tabbed dashboards.

## Prerequisites & Installation

Ensure you are running Python 3.8+ and have a virtual environment activated. 

Install the required dependencies:
```bash
pip install opencv-python numpy pillow cinereader pandas matplotlib
```
*(Note: Tkinter is typically included with standard Python installations, but Linux users may need to install `python3-tk` via their package manager).*

## Usage Guide

### Module 1: Video Analysis
1. Run the application: `python main.py`
2. **Load:** Click `1. Load .CINE File` and select your high-speed footage.
3. **Calibrate Parameters:**
    * Enter your optical scale (e.g., `1 mm = 50 pixels`).
    * Enter the initial pre-impact droplet diameter in mm (used for $d^*$ and $h^*$).
    * Navigate to the exact frame where the droplet touches the surface and click **Set Current Frame as t=0**.
4. **Set Baseline:** Right-click and drag the red dashed line to rest exactly on top of your substrate.
5. **Track:** Scrub through the video. A green bounding box will auto-detect the droplet. If it misses:
    * Left-click in empty space to draw a new box.
    * Click inside the box (cyan handle) to move it.
    * Click the edges/corners (yellow handles) to resize it.
    * Press `[ESC]` to delete your manual box and attempt auto-detection again.
6. **Export:** Click `2. Export & Plot Single Case`. Choose a save location. A `.csv` will be generated, and a dashboard will pop up showing your linear and log-scale dynamics.

### Module 2: Multi-Case Comparison
1. Switch to the **2. Multi-Case Comparison** tab in the main window.
2. Click `+ Load CSV Files` and select multiple `.csv` files you previously exported from Module 1.
3. Toggle the **Log Scale** checkbox if you want to focus exclusively on post-impact ($t > 0$) micro-dynamics.
4. Click `Generate Comparison Plot`. The software will automatically color-code and apply unique markers to overlay all selected cases on a single, publication-ready graph.

## Output Data Format

The exported `.csv` file retains full fidelity and contains the following columns for independent analysis in MATLAB, Origin, or Excel:
* `Frame`: Absolute frame index.
* `Time_s`: Precise hardware time in seconds (relative to impact $t=0$). Retains negative values for pre-impact.
* `Height_mm`: Physical height above baseline.
* `Diameter_mm`: Physical droplet width.
* `Height_nondim_h_star`: $h/D_0$
* `Diameter_nondim_d_star`: $d/D_0$ (Spreading Factor)