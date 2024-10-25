# face_detection.py
import cv2

class FaceAndObjectDetector:
    def __init__(self, cascade_path=None):
        # Inicializa o classificador Haar Cascade para detecção de faces
        if cascade_path is None:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise IOError("Não foi possível carregar o classificador Haar Cascade.")

    def detect_faces(self, frame, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30)):
        # Converte para escala de cinza para detecção de faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        return faces

    def detect_movement(self, prev_frame, current_frame, min_contour_area=500):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calcula a diferença entre os quadros e aplica um limite
        diff_frame = cv2.absdiff(prev_gray, current_gray)
        _, thresh = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

        # Encontra contornos e filtra contornos pequenos para evitar ruídos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        return filtered_contours

    def classify_scene(self, frame, prev_frame):
        faces = self.detect_faces(frame)

        # Se detectou faces, desenha retângulos ao redor das faces
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return "Face Detected"

        # Detecta movimento se não houver faces
        movement_contours = self.detect_movement(prev_frame, frame)
        if len(movement_contours) > 0:
            return "Movement Detected"
        
        return "No significant object detected"
