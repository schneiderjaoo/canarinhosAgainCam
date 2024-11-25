import cv2
import requests
import json
from datetime import datetime

class PersonAndFaceDetector:
    def __init__(self, face_cascade_path=None, body_cascade_path=None, profile_cascade_path=None, api_url="http://localhost:5001/api/passagem-apontamento"):
        # Configurações iniciais
        self.api_url = api_url  # URL da sua API
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') if face_cascade_path is None else cv2.CascadeClassifier(face_cascade_path)
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml') if body_cascade_path is None else cv2.CascadeClassifier(body_cascade_path)
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml') if profile_cascade_path is None else cv2.CascadeClassifier(profile_cascade_path)
        
        if self.face_cascade.empty() or self.body_cascade.empty() or self.profile_cascade.empty():
            raise IOError("Não foi possível carregar os classificadores Haar Cascade para detecção de rosto ou corpo.")
        
        self.tracker = None
        self.face_box = None

    def detect_faces(self, frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(gray_frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        return faces

    def register_entry(self, user_id):
        data = {
            "CodApontamento": 0,  # Aqui você pode definir um valor único ou gerar dinamicamente
            "Quantidade": 1,  # Quantidade de pessoas que entraram (sempre 1, conforme mencionado)
            "DataHora": datetime.now().isoformat(),  # Data e hora atual
            "LinhaOnibus": "Pomerode"  # Linha de ônibus
        }

        # Envia um POST para a API para registrar a entrada
        try:
            response = requests.post('https://localhost:5001/api/passagem-apontamento', json=data, verify=False)
            if response.status_code == 200:
                print("Entrada registrada com sucesso!")
            else:
                print(f"Erro ao registrar a entrada. Status Code: {response.status_code}")
        except Exception as e:
            print(f"Erro ao se comunicar com a API: {str(e)}")

    def classify_scene(self, frame):
        height, width = frame.shape[:2]
        entry_line_x = int(width * 0.75)
        exit_line_x = int(width * 0.25)

        cv2.line(frame, (entry_line_x, 0), (entry_line_x, height), (0, 255, 0), 2)
        cv2.line(frame, (exit_line_x, 0), (exit_line_x, height), (0, 0, 255), 2)

        if self.tracker is None:
            faces = self.detect_faces(frame)
            if len(faces) > 0:
                self.face_box = faces[0]
                self.tracker = cv2.TrackerCSRT_create()
                self.tracker.init(frame, tuple(self.face_box))

        if self.tracker is not None:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                movement_status = "Rosto em movimento"
                if x < exit_line_x:
                    movement_status = "Entrada detectada"
                    cv2.putText(frame, "Entrada", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Quando detecta a entrada, registra na API
                    self.register_entry(user_id=123)  # Substitua 123 por um identificador real do usuário

                elif x > entry_line_x:
                    movement_status = "Saída detectada"
                    cv2.putText(frame, "Saída", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                movement_status = "Rosto perdido"

            if not success:
                self.tracker = None

        else:
            movement_status = "Sem rosto detectado"
        
        return movement_status
