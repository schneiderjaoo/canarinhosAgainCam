import cv2
from camera import Camera
from face_detection import PersonAndFaceDetector

def main():
    camera = Camera()
    detector = PersonAndFaceDetector()

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Erro ao capturar frame. Encerrando...")
            break

        classification = detector.classify_scene(frame)
        print(classification)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
