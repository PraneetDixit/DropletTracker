import cv2
import numpy as np

def detect_droplet(frame_8bit, baseline_y, threshold_val=100):
    """
    Detects the bounding box of a droplet above a specified baseline.
    """
    # Crop the image strictly above the baseline so reflections/plate are ignored
    roi = frame_8bit[0:max(0, baseline_y - 2), :]
    if roi.size == 0: 
        return None

    # Apply threshold based on the UI slider
    _, thresh = cv2.threshold(roi, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations to clean up sensor noise and close holes
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return None

    # Assume the largest contour is the droplet
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Ignore microscopic specks
    if cv2.contourArea(largest_contour) < 10: 
        return None 

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, w, h)