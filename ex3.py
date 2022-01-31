import cv2
import numpy as np
from matplotlib import pyplot as plt

directory = 'photos3/picture.jpg'

img = cv2.imread(directory)
imgGray = cv2.imread(directory, 0)

EqualizedHist = cv2.equalizeHist(imgGray)
ResultingImg = np.hstack((imgGray, EqualizedHist))
cv2.imwrite("photos3/EqualizedHist.jpg", ResultingImg)

Clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imwrite("photos3/example.jpg", lab_img)

l, a, b = cv2.split(lab_img)
l = Clahe.apply(l)
lab_img = cv2.merge([l, a, b])
back_to_rgb = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
ResultingImgLocalOrigLabRgb = np.hstack((img, lab_img, back_to_rgb))
cv2.imwrite("photos3/ResultingImgLocalOrigLabRgb.jpg",  ResultingImgLocalOrigLabRgb)

cl1 = Clahe.apply(imgGray)
ResultingImgLocalGray = np.hstack((imgGray, cl1))
cv2.imwrite("photos3/LocalizedEqualizedHistGray.jpg", ResultingImgLocalGray)

GaussianBlurImg = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
cv2.imwrite("photos3/GaussianBlur.jpg", GaussianBlurImg)

SobelImgWithoutBlur = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)
cv2.imwrite("photos3/SobelImgWithoutBlur.jpg", SobelImgWithoutBlur)

SobelImgWithBlur = cv2.Sobel(GaussianBlurImg, cv2.CV_64F, 1, 0, ksize=5)
cv2.imwrite("photos3/SobelImgWithBlur.jpg", SobelImgWithBlur)

LaplacianImgWithoutBlur = cv2.Laplacian(imgGray, cv2.CV_64F)
cv2.imwrite("photos3/LaplacianImgWithoutBlur.jpg", LaplacianImgWithoutBlur)

LaplacianImgWithBlur = cv2.Laplacian(GaussianBlurImg, cv2.CV_64F)
cv2.imwrite("photos3/LaplacianImgWithBlur.jpg", LaplacianImgWithBlur)

