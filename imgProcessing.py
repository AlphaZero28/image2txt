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
                img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # erosion and dialation applied
        # kernel_size = [(3, 3, 3), (5, 5, 5)]
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # img = cv2.convertScaleAbs(img, alpha=1)

        # gauss = cv2.GaussianBlur(img, (11, 11), 10)
        # img = cv2.addWeighted(img, 2.5, gauss, -1, 0)
        # img = np.roll(img, 2)
        return img

    def horizontal_hist(self, img):
        ''' calculate horizontal histogram of the image '''

        projection = np.sum(img, 1) / 255              # histogram data
        result = np.zeros((projection.shape[0], img.shape[1]))  # img data
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
        ''' Find lines that exist in the image'''

        cropped_images = []
        for i, (r1, r2) in enumerate(bounding_horizontal_rect):
            crop_img = img[r1:r2, 0:img.shape[1]]
            cropped_images.append(crop_img)

        return cropped_images

    def vertical_hist(self, img):
        ''' meant to be used after horizontal histogram in each line'''
        projection = np.sum(img, 0) / 255
        result = np.zeros((img.shape[0], projection.shape[0]))

        for col in range(img.shape[1]):
            x2 = int(projection[col])
            cv2.line(result, (col, 0), (col, x2), (255, 255, 255), 1)

        return [result, projection]

    def bounding_vertical_rect(self, vert_data):
        ''' returns the initial and final x axis point of each word'''
        bounding_vertical_rect = []
        valp = 0
        pos1 = 0
        thresh_val = 0

        for i, val in enumerate(vert_data):
            if val > thresh_val and valp <= thresh_val:
                pos1 = i-1

            elif (not i == 0) and val <= thresh_val and valp > thresh_val:
                if (i-pos1) > 2:
                    bounding_vertical_rect.append((pos1, i+1))
            valp = val

        return bounding_vertical_rect

    def find_words(self, bounding_vertical_rect, img):
        ''' Find words in the line image'''
        cropped_images = []
        for i, (r1, r2) in enumerate(bounding_vertical_rect):
            crop_img = img[0:img.shape[1], r1:r2]
            cropped_images.append(crop_img)

        return cropped_images

    def pytesseract_apply(self, img, flag):
        ''' Text detection in the image. 
        flag = 0 when image conatains line text. 
        flag = 1 when image containes word text '''

        if flag == 0:
            set_config = '--psm 7'
        elif flag == 1:
            set_config = '--psm 6'
        # im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        im2 = np.pad(img, ((400, 400), (200, 200)), 'constant',
                     constant_values=(255, 255))

        text = pytesseract.image_to_string(
            im2, lang='ben', config=set_config)
        return text
