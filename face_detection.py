import cv2
import numpy as np 
import requests
import time
import threading
from datetime import datetime

class PersonAndFaceDetector:
    def __init__(self, api_url="https://localhost:5001/api/passagem-apontamento", bus_line=""):
        self.api_url = api_url
        self.bus_line = bus_line
        self.failed_requests = []  # Fila para requisições falhadas

        self.face_net = cv2.dnn.readNetFromCaffe(
            'models/deploy.prototxt',
            'models/res10_300x300_ssd_iter_140000.caffemodel'
        )

        self.tracker = None
        self.last_event_time = 0
        self.event_cooldown = 3  # Intervalo mínimo entre chamadas da API
        self.entry_detected = False
        self.exit_detected = False
        
        # Inicia o processo de reenvio em uma thread separada
        self.start_retry_thread()

    def start_retry_thread(self):
        """Thread para reenviar requisições falhadas."""
        retry_thread = threading.Thread(target=self.retry_requests, daemon=True)
        retry_thread.start()

    def retry_requests(self):
        """Tenta reenviar as requisições falhadas periodicamente."""
        while True:
            if self.failed_requests:
                print("Tentando reenviar requisições falhadas...")
                for request_data in self.failed_requests[:]:
                    try:
                        response = requests.post(self.api_url, json=request_data, verify=False)
                        if response.status_code == 200:
                            print("Requisição reenviada com sucesso!")
                            self.failed_requests.remove(request_data)
                        else:
                            print(f"Falha no reenvio. Status Code: {response.status_code}")
                    except Exception as e:
                        print(f"Erro ao reenviar requisição: {str(e)}")
            time.sleep(10)  # Verifica a fila a cada 10 segundos

    def detect_faces_dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # Confiança mínima
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                return (startX, startY, endX - startX, endY - startY)
        return None

    def register_entry(self, is_entry):
        data = {
            "CodApontamento": 0,
            "Quantidade": 1,
            "DataHora": datetime.now().isoformat(),
            "LinhaOnibus": self.bus_line,
            'EhEntrada': is_entry
        }

        try:
            response = requests.post(self.api_url, json=data, verify=False)
            if response.status_code == 200:
                print("Entrada registrada com sucesso!")
            else:
                print(f"Erro ao registrar a entrada. Status Code: {response.status_code}")
                self.failed_requests.append(data)  # Adiciona à fila em caso de erro
        except Exception as e:
            print(f"Erro ao se comunicar com a API: {str(e)}")
            self.failed_requests.append(data)  # Adiciona à fila em caso de falha de conexão

    def classify_scene(self, frame):
        height, width = frame.shape[:2]
        entry_line_x = int(width * 0.75)
        exit_line_x = int(width * 0.25)

        cv2.line(frame, (entry_line_x, 0), (entry_line_x, height), (0, 255, 0), 2)
        cv2.line(frame, (exit_line_x, 0), (exit_line_x, height), (0, 0, 255), 2)

        if self.tracker is None:
            face_box = self.detect_faces_dnn(frame)
            if face_box:
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(frame, tuple(face_box))
        else:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                current_time = time.time()
                if x < exit_line_x and not self.entry_detected:
                    if current_time - self.last_event_time > self.event_cooldown:
                        self.register_entry(is_entry=True)
                        self.last_event_time = current_time
                        self.entry_detected = True
                        self.exit_detected = False

                elif x > entry_line_x and not self.exit_detected:
                    if current_time - self.last_event_time > self.event_cooldown:
                        self.register_entry(is_entry=False)
                        print("Saída detectada!")
                        self.last_event_time = current_time
                        self.exit_detected = True
                        self.entry_detected = False
            else:
                self.tracker = None
                self.entry_detected = False
                self.exit_detected = False

        return frame
