import cv2
import numpy as np

def detect_droplet(frame_8bit, baseline_y):
    """
    Analyzes an 8-bit frame to find the droplet bounding box.
    Uses adaptive thresholding and smart filtering to ignore the substrate.
    """
    # 1. Isolate Region of Interest (ROI)
    # We subtract 5 pixels from the baseline so the script physically 
    # cannot see the dark substrate line, preventing false positives.
    safe_baseline = max(0, int(baseline_y) - 5)
    
    if safe_baseline < 10: 
        return None # Baseline is too high up to detect anything

    roi = frame_8bit[:safe_baseline, :]
    
    # 2. Blur to remove high-speed sensor noise (salt and pepper)
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)

    # 3. Adaptive Thresholding
    # Excellent for uneven lighting. Inverts so the dark droplet becomes white.
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )

    # 4. Morphological Closing (Fill holes in the droplet mask)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 5. Contour Extraction
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 6. Smart Filtering
    valid_contours = []
    max_area = roi.shape[0] * roi.shape[1] * 0.8 # 80% of the screen
    
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        
        # Ignore tiny specks (< 50px) and massive background shadows
        if 50 < area < max_area:
            valid_contours.append((c, area, x, y, w, h))

    if not valid_contours:
        return None

    # Assume the droplet is the largest valid object remaining
    largest_contour_data = max(valid_contours, key=lambda item: item[1])
    _, _, x, y, w, h = largest_contour_data

    return (x, y, w, h)