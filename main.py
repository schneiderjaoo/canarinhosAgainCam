import cv2
from camera import Camera
from face_detection import PersonAndFaceDetector

def main():
    camera = Camera()
    detector = PersonAndFaceDetector()
    prev_positions = None

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Erro ao capturar frame. Encerrando...")
            break

        direction, prev_positions = detector.classify_scene(frame, prev_positions)
        print(direction)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
