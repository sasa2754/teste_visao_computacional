import numpy as np
import matplotlib
import cv2 as cv
import sys

url = "./videoMao.mp4"

cap = cv.VideoCapture(url)

while True:
    ret, frame = cap.read()

    if not ret:
        # print("Falha ao capturar vídeo")
        print("Fim do vídeo")
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    limiteInferior = np.array([0, 48, 80])  
    limiteSuperior = np.array([20, 255, 255])  


    mask = cv.inRange(hsv, limiteInferior, limiteSuperior)

    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour) > 500:
            cv.drawContours(frame, [contour], -1, (0, 255, 0), 3)

            M = cv.moments(contour)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                cv.putText(frame, "Sabrina", (cX - 20, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv.imshow("Simulação do vídeo", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()