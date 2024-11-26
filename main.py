import cv2
from camera import Camera
from face_detection import PersonAndFaceDetector

def main():
    camera = Camera()
    
    # Dicionário com as opções de linhas de ônibus
    lines = {
        '1': 'Jaraguá do Sul',
        '2': 'Pomerode',
        '3': 'Corupá',
        '4': 'Guaramirim',
        '5': 'Schroeder',
        '6': 'Barra Velha'
    }
    
    detection_enabled = False  # Controle para iniciar/parar a detecção
    detector = None  # Inicializa o detector
    
    print("Tecle 1 para iniciar a detecção e contagem.")
    
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Erro ao capturar frame. Encerrando...")
            break
        
        key = cv2.waitKey(1) & 0xFF
        
        if not detection_enabled and key == ord('1'):
            # Menu para escolher a linha de ônibus
            print("\nEscolha a linha de ônibus:")
            for key_line, line in lines.items():
                print(f"{key_line} - {line}")

            # Captura a escolha do usuário
            selected_key = input("Digite o número correspondente à linha: ")
            selected_line = lines.get(selected_key)

            if not selected_line:
                print("Opção inválida! Tente novamente.")
                continue  # Retorna ao menu se a entrada for inválida

            print(f"Linha selecionada: {selected_line}")
            detector = PersonAndFaceDetector(bus_line=selected_line)  # Atualiza o detector
            print("Detecção iniciada. Tecle 2 para finalizar.")
            detection_enabled = True
        
        elif detection_enabled and key == ord('2'):
            print("Detecção finalizada. Tecle 1 para reiniciar.")
            detection_enabled = False
        
        if detection_enabled:
            classification = detector.classify_scene(frame)
            cv2.imshow('Frame', classification)
        else:
            cv2.imshow('Frame', frame)
        
        if key == ord('q'):
            print("Encerrando o programa...")
            break
    
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
