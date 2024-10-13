# main.py
import cv2
from camera import Camera
from face_detection import FaceAndObjectDetector

def main():
    camera = Camera()  # Inicializa a câmera
    detector = FaceAndObjectDetector()

    prev_frame = camera.get_frame()  # Pega o primeiro frame

    while True:
        frame = camera.get_frame()
        if frame is None:
            break

        # Classifica a cena com base no frame atual e no anterior
        classification = detector.classify_scene(frame, prev_frame)
        print(classification)

        # Exibe o frame atual com as detecções
        cv2.imshow('Frame', frame)

        # Atualiza o frame anterior para o próximo ciclo
        prev_frame = frame.copy()

        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
