import numpy as np
import matplotlib
import cv2 as cv
import sys

url = "./videoMao.mp4"

cap = cv.VideoCapture(url) #Cria um objeto de captura de vídeo para ler quadro a quadro

while True:
    # ret é um valor booleano que diz se a leitura foi bem-sucedida, frame é o quadro atual do vídeo, a função cap.read() retorna esses dois valores
    ret, frame = cap.read()

    if not ret:
        # print("Falha ao capturar vídeo")
        print("Fim do vídeo")
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) #Converte o quadro de sua cor original (BGR) para o formato HSV, para facilitar a segmentação de objetos por cor.

    # Define os limetes da cor no formato hsv, o primeiro controla a tonalidade, o segundo controla a saturação, e o terceiro controla o valor
    limiteInferior = np.array([0, 48, 80])  
    limiteSuperior = np.array([20, 255, 255])  


    # Cria uma máscara binária, onde todos os pixels que se encaixam no aspectro entre os dois limites são convertidos em branco (255), e todos os outros em preto (0)
    mask = cv.inRange(hsv, limiteInferior, limiteSuperior)


    # Encontra os contornos na máscara binária, e retorna os valores dos contornos em uma lista
    # O primeiro parâmetro é a máscara onde os contornos serão detectados
    # O segundo parâmetro (cv.RETR_TREE), é um método que recupera os contornos em uma hierarquia (é útil para quando existem objetos dentro de outros)
    # O terceiro parâmetro (cv.CHAIN_APPROX_SIMPLE), é o método que comprime os contornos para que ocupem menos memória
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Iteração para cada contorno encontrado
    for contour in contours:
        if cv.contourArea(contour) > 500: # Verifica se a área de contorno é maior que 500 pixels, para filtrar objetos pequenos os ruídos do vídeo, que podem ter sido detectador sem querer
            #Desenha o contorno encontrado no quadro original (frame) com a cor verde (0, 255, 0) e uma espessura de 3 pixels
            cv.drawContours(frame, [contour], -1, (0, 255, 0), 3)

            # Calcula os momentos geométricos do contorno, esses momentos podem ser usados para calcular várias propriedades de uma forma, como área, centro de massa, etc
            M = cv.moments(contour)

            # Verifica se o momento de ordem 0 não é 0, o que significaria que há uma área válida
            if M["m00"] != 0:
                # Calcula a coordenada X e Y do centro de massa do objeto detectado
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Desenha um círculo branco no centro do objeto detectado, o círculo tem um raio de 7 pixels
                cv.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                # Escreve algo acima do circulo, o texto é exibido em branco
                cv.putText(frame, "Sabrina", (cX - 20, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Exibe o vídeo com os contornos e centro de massa desenhados em uma janela que pode ser nomeada
    cv.imshow("Simulação do vídeo", frame)

    # Aguarda 1 milissegundo por uma tecla pressionada, se a tecla pressionada for q, o programa encerra o loop e fecha a janela
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o objeto de captura do vídeo, fechando o arquivo
cap.release()
# Fecha todas as janelas abertas pelo openCV
cv.destroyAllWindows()