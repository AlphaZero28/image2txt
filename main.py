import pytesseract
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from imgProcessing import imgProcess

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

imgProcess = imgProcess()
img = cv2.imread('D:\doc-app\images\sonar-tori.png')


def img2txt(img):
    img_gray = imgProcess.bgr2gray(img)
    img_inv = imgProcess.invertImage(img_gray)
    text = imgProcess.pytesseract_apply(img_inv)
    return text


# plt.imshow(img_inv, cmap='gray')
# plt.axis('off')
# plt.show()


text = img2txt(img)

fh = open('img-text.txt', 'w', encoding="utf-8")
fh.write(text)
fh.close()
