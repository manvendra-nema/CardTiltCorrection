import cv2
import numpy as np

def correct_tilt(image):
    """
    Corrects the tilt of an ID card in an image.
    
    Parameters:
        image (numpy.ndarray): Input image containing a tilted ID card.
    
    Returns:
        numpy.ndarray: Deskewed image with corrected orientation.
    """
    height, width = image.shape[:2]
    border_margin = 10  # Exclude lines near image boundaries
    
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80,
                            minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return image  # Return original if no lines are found
    
    # Filter lines: remove short lines and those touching image boundaries
    min_length = min(width, height) * 0.25  # 25% of smaller image dimension
    valid_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (x1 < border_margin or x1 > width - border_margin or
            y1 < border_margin or y1 > height - border_margin or
            x2 < border_margin or x2 > width - border_margin or
            y2 < border_margin or y2 > height - border_margin):
            continue
        
        length = np.hypot(x2 - x1, y2 - y1)
        if length >= min_length:
            valid_lines.append((line[0], length))
    
    if not valid_lines:
        return image  # Return original if no valid lines are found
    
    # Pick the longest line
    (longest_line, _) = max(valid_lines, key=lambda x: x[1])
    x1, y1, x2, y2 = longest_line
    
    # Compute the tilt angle
    angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
    
    # Adjust for vertical edges
    if abs(angle) > 45:
        correction_angle = angle - 90 if angle > 0 else angle + 90
    else:
        correction_angle = angle
    
    # Rotate image to deskew
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, -correction_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated
