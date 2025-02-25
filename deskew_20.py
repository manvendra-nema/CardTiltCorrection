import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_perspective_distortion(image, template):
    """
    Corrects perspective distortion of the input image using SIFT feature matching
    and RANSAC. The computed homography maps points from the template (ideal frontal view)
    to the input image. The inverse homography is used to warp the image into the template's
    coordinate system, yielding a corrected, cropped output.
    
    Parameters:
        image (numpy.ndarray): Input image with perspective distortion.
        template (numpy.ndarray): Reference template image (with ideal perspective).
    
    Returns:
        tuple: (warped_image, homography matrix, projected_corners)
            - warped_image: the corrected, deskewed, and cropped image.
            - homography: the 3x3 homography matrix computed from template→image.
            - projected_corners: the four corners of the template mapped onto the input image.
    """
    # Convert images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    kp_template, des_template = sift.detectAndCompute(gray_template, None)
    kp_image, des_image = sift.detectAndCompute(gray_image, None)
    
    # Use FLANN-based matcher to find feature matches
    index_params = dict(algorithm=1, trees=5)  # Using KDTree (algorithm=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des_template, des_image, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        print("Not enough good matches were found to compute homography.")
        return image, None, None
    
    # Extract matched keypoint coordinates
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Compute the inverse homography to warp the input image into the template's coordinate system.
    H_inv = np.linalg.inv(H)
    h_temp, w_temp = template.shape[:2]
    warped_image = cv2.warpPerspective(image, H_inv, (w_temp, h_temp),
                                       flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Optionally, compute the template's corners projected into the input image.
    template_corners = np.float32([[0, 0],
                                   [w_temp, 0],
                                   [w_temp, h_temp],
                                   [0, h_temp]]).reshape(-1, 1, 2)
    projected_corners = cv2.perspectiveTransform(template_corners, H)
    
    return warped_image, H, projected_corners

# Example usage:
template = cv2.imread(r"addhar_temp.jpg")  # Provide your reference template image (correct orientation)
image = cv2.imread(r".\20th Feb\1000279565.jpg")     # Image to be deskewed
if template is None or image is None:
    raise ValueError("Template and/or input image could not be loaded. Check the file paths.")

# Perform perspective distortion correction.
corrected_image, homography, projected_corners = correct_perspective_distortion(image, template)

# Visualize the results.
# 1. Draw the detected template corners on the input image.
image_with_corners = image.copy()
if projected_corners is not None:
    projected_corners_int = np.int32(projected_corners)
    cv2.polylines(image_with_corners, [projected_corners_int], True, (0, 255, 0), 3)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
plt.title("Input Image with Projected Template Corners")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
plt.title("Perspective Corrected Image")
plt.axis("off")

plt.tight_layout()
plt.show()
