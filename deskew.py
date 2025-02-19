import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load the image (update the filename as needed)
image = cv2.imread("id_card.jpg")
if image is None:
    raise ValueError("Image not found. Check your path.")

orig = image.copy()
height, width = image.shape[:2]
border_margin = 10  # pixels to exclude near image boundaries

# Convert to grayscale and blur to reduce noise
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Detect line segments using HoughLinesP
lines = cv2.HoughLinesP(edges,
                        rho=1,
                        theta=np.pi/180,
                        threshold=80,
                        minLineLength=50,
                        maxLineGap=10)

if lines is None:
    raise ValueError("No lines detected. Try adjusting Hough parameters.")

# Filter out lines that are too short (likely text) or touch image boundaries.
min_length = min(width, height) * 0.25  # adjust as needed (25% of the smaller image dimension)
valid_lines = []

for line in lines:
    x1, y1, x2, y2 = line[0]
    # Skip lines touching the image boundary
    if (x1 < border_margin or x1 > width - border_margin or
        y1 < border_margin or y1 > height - border_margin or
        x2 < border_margin or x2 > width - border_margin or
        y2 < border_margin or y2 > height - border_margin):
        continue

    # Calculate line length
    length = np.hypot(x2 - x1, y2 - y1)
    if length < min_length:
        continue  # Skip short lines (likely text or small details)
    
    valid_lines.append((line[0], length))

if not valid_lines:
    raise ValueError("No valid lines found after filtering.")

# Choose the longest valid line (assumed to correspond to an edge of the ID card)
(longest_line, longest_length) = max(valid_lines, key=lambda x: x[1])
x1, y1, x2, y2 = longest_line

# Compute the angle of this line (in degrees) relative to horizontal
angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
print("Longest line angle:", angle)

# If the line is nearly vertical, adjust it to obtain the horizontal tilt.
# (If the line is nearly vertical, its angle is close to 90° or -90°.
#  We subtract/add 90° to obtain the corresponding tilt relative to horizontal.)
if abs(angle) > 45:
    correction_angle = angle - 90 if angle > 0 else angle + 90
else:
    correction_angle = angle

print("Correction angle:", correction_angle)

# Rotate the image by the negative of the correction angle to deskew
center = (width // 2, height // 2)
M = cv2.getRotationMatrix2D(center, -correction_angle, 1.0)
rotated = cv2.warpAffine(orig, M, (width, height),
                         flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)

# For visualization: draw the detected longest line on the original image
cv2.line(orig, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Display the results using cv2_imshow (Google Colab)
cv2_imshow(orig)      # Original image with longest line drawn
cv2_imshow(rotated)   # Deskewed image
