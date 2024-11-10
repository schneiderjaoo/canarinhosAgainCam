import cv2

class PersonAndFaceDetector:
    def __init__(self, face_cascade_path=None, body_cascade_path=None):
        if face_cascade_path is None:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        if body_cascade_path is None:
            self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        else:
            self.body_cascade = cv2.CascadeClassifier(body_cascade_path)

        if self.face_cascade.empty() or self.body_cascade.empty():
            raise IOError("Não foi possível carregar os classificadores Haar Cascade para detecção de rosto ou corpo.")

    def detect_faces(self, frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        return faces

    def detect_body(self, frame, face_area, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = face_area
        body_area = frame[y + h: , x: x + w]

        bodies = self.body_cascade.detectMultiScale(body_area, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        bodies = [(x + bx, y + h + by, bw, bh) for (bx, by, bw, bh) in bodies]
        return bodies

    def classify_scene(self, frame):
        faces = self.detect_faces(frame)
        bodies = []

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) 

            for (x, y, w, h) in faces:
                body = self.detect_body(frame, (x, y, w, h))
                bodies.extend(body)

            return "Face Detected"

        if len(bodies) > 0:
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde para corpo
            return "Body Detected"

        return "No person detected"
