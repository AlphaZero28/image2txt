import cv2
import numpy as np
import PIL.Image as Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class imgProcess():

    def bgr2gray(self, img):
        ''' convert to gray image '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def invertImage(self, img):
        ''' invert pixels if 255 is dominant'''

        if (img.sum(axis=1).sum()/img.size) > 50:
            # img = 255- img
            ret, img = cv2.threshold(
                img, 0, 255, cv2.THRESH_OTSU)

        return img

    def horizontal_hist(self, img):
        ''' calculahorizontal histogram of the image '''

        projection = np.sum(img, 1) / 255
        result = np.zeros((projection.shape[0], img.shape[1]))
        for row in range(img.shape[0]):
            # /max(projection)*img.shape[1]
            x2 = int(projection[row])
            cv2.line(result, (0, row), (x2, row), (255, 255, 255), 1)

        return [result, projection]

    def bounding_horizontal_rect(self, hist_data):
        ''' returns the initial and final y axis point of each line'''
        bounding_horizontal_rect = []
        valp = 0
        pos1 = 0
        thresh_val = 0

        for i, val in enumerate(hist_data):
            if val > thresh_val and valp <= thresh_val:
                pos1 = i-1

            elif (not i == 0) and val <= thresh_val and valp > thresh_val:
                if (i-pos1) > 2:
                    bounding_horizontal_rect.append((pos1, i+1))
            valp = val

        return bounding_horizontal_rect

    def find_lines(self, bounding_horizontal_rect, img):
        cropped_images = []
        for i, (r1, r2) in enumerate(bounding_horizontal_rect):
            crop_img = img[r1:r2, 0:img.shape[1]]
            cropped_images.append(crop_img)

        return cropped_images

    def pytesseract_apply(self, img):
        # im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im2 = np.pad(img, ((500, 500), (500, 500)), 'constant',
                     constant_values=(0, 0))

        text = pytesseract.image_to_string(
            im2, lang='ben', config='--psm 11')
        return text
