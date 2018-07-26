import cv2
import numpy

# Load two images
img1 = cv2.imread('person.jpg')
img2 = cv2.imread('orange.png')
# I want to put logo on top-left corner, So I create a ROI
img1 = cv2.resize(img1, (600, 400))
rows, cols, channels = img2.shape
print("{}, {}, {}".format(rows, cols, channels))
roi = img1[0:rows, 0:cols]
# Now create a mask of logo and create its inverse mask also
img2gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow('img2gray_1', img2gray_1)
cv2.imshow('img2gray_2', img2gray_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret1, mask1 = cv2.threshold(img2gray_1, 10, 255, cv2.THRESH_BINARY)
ret2, mask2 = cv2.threshold(img2gray_2, 10, 255, cv2.THRESH_BINARY)

cv2.imshow('mask1', mask1)
cv2.imshow('mask2', mask2)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask_inv_1 = cv2.bitwise_not(mask1)
mask_inv_2 = cv2.bitwise_not(mask2)

cv2.imshow('mask_inv_1', mask_inv_1)
cv2.imshow('mask_inv_2', mask_inv_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(img1, img1, mask=mask1)
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask2)

cv2.imshow('img1_bg', img1_bg)
cv2.imshow('img2_fg', img2_fg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Put logo in ROI and modify the main image
dst = cv2.add(roi, img2_fg)
img1[0:rows, 0:cols] = dst
cv2.imshow('Fin', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
