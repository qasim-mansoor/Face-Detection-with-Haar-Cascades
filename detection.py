import cv2 as cv

img = cv.imread("Photos/group 1.jpg")
cv.imshow("Lady", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray)

haar = cv.CascadeClassifier("haar_face.xml")

faces_rect = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(len(faces_rect))

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

cv.imshow("Detected faces", img)

cv.waitKey(0)
