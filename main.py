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
    # conversion to grayscale
    img_gray = imgProcess.bgr2gray(img)
    # invert the image
    img_inv = imgProcess.invertImage(img_gray)

    # horizontal histogram applied to find lines
    [hist_img, hist_data] = imgProcess.horizontal_hist(img_inv)
    bounding_horizontal_rect = imgProcess.bounding_horizontal_rect(hist_data)
    lines = imgProcess.find_lines(bounding_horizontal_rect, img_gray)

    line_str = []
    final_text = []
    for i, line in enumerate(lines):
        # OCR applied on the each line
        text = imgProcess.pytesseract_apply(line, 0)

        # if text not founnd at all segment line image to word image
        if len(text) == 0:
            [vert_img, vert_data] = imgProcess.vertical_hist(
                imgProcess.invertImage(line))
            bounding_vertical_rect = imgProcess.bounding_vertical_rect(
                vert_data)
            words = imgProcess.find_words(bounding_vertical_rect, line)

            # OCR applied on each word
            for word in words:
                text = imgProcess.pytesseract_apply(word, 1)

                line_str.append(text)

            text = " ".join(line_str)
        text = text.replace("\n", "")
        final_text.append(text)
        line_str.clear()
    return final_text


text = img2txt(img)

print(text)

# plt.imshow(words[0], cmap='gray')
# plt.axis('off')
# plt.show()
