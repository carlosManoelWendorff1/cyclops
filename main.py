import cv2
import numpy as np  # Importe numpy para a manipulação de arrays NumPy

# Carregue o modelo MobileNet SSD pré-treinado
net = cv2.dnn.readNetFromCaffe(
    'MobileNetSSD_deploy.prototxt',  # Arquivo de configuração
    'MobileNetSSD_deploy.caffemodel'  # Modelo treinado
)

# Defina as classes de objetos que o modelo pode detectar
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Inicialize a câmera
cap = cv2.VideoCapture(0)  # Use 0 para a câmera padrão, ou especifique um número de dispositivo se necessário

while True:
    # Capture um quadro da câmera
    ret, frame = cap.read()

    # Realize a detecção de objetos
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300), (127.5, 127.5, 127.5)))
    net.setInput(blob)
    detections = net.forward()

    # Itere sobre as detecções e desenhe caixas delimitadoras
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >0.65:  # Defina um limite de confiança apropriado
            class_id = int(detections[0, 0, i, 1])
            class_name = CLASSES[class_id]
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)

            # Desenhe a caixa delimitadora e o nome da classe
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, class_name, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exiba o quadro com as detecções
    cv2.imshow("Detecção de Objetos", frame)

    # Saia do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere a captura e feche a janela
cap.release()
cv2.destroyAllWindows()
