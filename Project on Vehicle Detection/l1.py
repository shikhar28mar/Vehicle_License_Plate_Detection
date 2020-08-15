import imutils
import cv2
import pytesseract as tess
tess.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image=cv2.imread('4.jpeg')
image=imutils.resize(image, width=500)

cv2.imshow("Original Image",image)
cv2.waitKey(0)

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("1 - Grayscale Conversion", gray)
cv2.waitKey(0)

gray= cv2.bilateralFilter (gray, 11, 17, 17)
cv2.imshow("2 - Bilateral Filter", gray)
cv2.waitKey(0)

edged = cv2.Canny(gray, 170, 200)
cv2.imshow("3 - Canny Edged", edged)
cv2.waitKey(0)


cnts, new = cv2.findContours(edged.copy() , cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


img1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
cv2.imshow("4- All Contours", img1)
cv2.waitKey(0)

cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:50]
NumberPlateCnt = None

img2 = image.copy()
cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
cv2.imshow("5- Top 50 Contours", img2)
cv2.waitKey(0)


count = 0
idx = 7
for c in cnts:
       peri = cv2.arcLength(c, True)
       approx = cv2.approxPolyDP(c, 0.02 * peri, True)
       print("approx = ", approx)
      
       NumberPlateCnt = approx #this is approx Number plate Countour
       x, y, w, h =cv2.boundingRect(c)
       new_img = image[y:y +h, x:x + w]
       cv2.imwrite('lp3/' + str(idx) + '.jpeg', new_img)
       idx+=1
       
              


cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.imshow("Final Image With Number Plate Detected", image)
cv2.waitKey(0)

Cropped_img_loc = 'lp3/7.jpeg'
cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))

text = tess.image_to_string(Cropped_img_loc, lang='eng')
print("Number is :",text)

cv2.waitKey(0)
            
       

