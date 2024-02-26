import cv2 
import pytesseract
import numpy as np
import sys


def ocr_core(img):
    """This functions handles OCR preprocessing of images"""
    # here we'll use pillow images class to open the image and pytesseract to detect 
    
    # custom_config = r'--oem 3 --psm 6'
    text=pytesseract.image_to_string(img)
    return text
# pass file path from command line using sys.argv.
# image_path = sys.argv[1]
img = cv2.imread('ocr-scan-example.png')
# img = cv2.imread(sys.argv[0])


# if img is not None:
#     print("Image loaded success fully.")
# else:
#     print("Error loading image.")

# ocr engine mode and page segmentation mode
# custom_config = r'--oem 3 --psm 6'
# pytesseract.image_to_string(img, config=custom_config)

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding of image
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation of image
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion of img
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 


img = get_grayscale(img)
img = remove_noise(img)
img = thresholding(img)
# img = opening(img)
# img = deskew(img)


print(ocr_core(img))
