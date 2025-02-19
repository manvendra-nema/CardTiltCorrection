# 🚀 Overview

Ever scanned an ID card only to find it slightly tilted? Correct Tilt is here to fix that! Using edge detection and line analysis, this function identifies the main edges of the card and rotates it back to its proper alignment.

# 🔍 How It Works

Converts the image to grayscale and applies Gaussian blur.

Detects edges using Canny edge detection.

Finds significant lines using a customized Hough Transform approach.

Selects the longest perpendicular edge to estimate the tilt angle.

Rotates the image to correct the tilt.

# 🛠 Usage

from google.colab.patches import cv2_imshow

 Load the image
image = cv2.imread("id_card.jpg")

corrected_image = correct_tilt(image)

 Display the corrected image

cv2_imshow(corrected_image)

# 📌 Features

✅ Automatic tilt correction based on image content✅ Ignores small text lines to focus on card edges✅ Avoids falsely detecting image boundaries as edges✅ Works on cropped images with partial ID cards

# 🎯 Why Use It?

Saves manual effort in aligning ID cards

Enhances OCR accuracy by ensuring properly aligned text

Lightweight and efficient with minimal dependencies

📜 License

Feel free to use, modify, and improve! Contributions are welcome. 🚀

