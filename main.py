import cv2
from camera import Camera
from face_detection import FaceDetector

def main():
    try:
        camera = Camera()
    except ValueError as e:
        print(e)
        return

    try:
        detector = FaceDetector()
    except IOError as e:
        print(e)
        camera.release()
        return

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Não foi possível capturar o frame da câmera.")
            break

        faces = detector.detect_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Detecção de Rostos', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
