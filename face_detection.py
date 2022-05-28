import cv2
import numpy as np
import face_recognition

imgshelly = face_recognition.load_image_file('sample_images/shelly.jpg')
imgshelly = cv2.cvtColor(imgshelly, cv2.COLOR_BGR2RGB)
imgshellytest = face_recognition.load_image_file('sample_images/jordan.jpg')
imgshellytest = cv2.cvtColor(imgshellytest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgshelly)[0]
encodeshelly = face_recognition.face_encodings(imgshelly)[0]
cv2.rectangle(imgshelly, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgshellytest)[0]
encodeshellytest = face_recognition.face_encodings(imgshellytest)[0]
cv2.rectangle(imgshellytest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeshelly], encodeshellytest)
faceDis = face_recognition.face_distance([encodeshelly], encodeshellytest)
print(results, faceDis)
cv2.putText(imgshellytest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('shelly', imgshelly)
cv2.imshow('shelly_test', imgshellytest)
cv2.waitKey(0)