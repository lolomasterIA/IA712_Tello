import cv2
import os
import time
import cvlib as cv
from cvlib.object_detection import draw_bbox
from ultralytics import YOLO


def scan_for_human_cvlib(tello, verbose=True):
    """
    Effectue une rotation à 360°, sauvegarde les frames + résultats
    Retourne (True, frame) si une personne est détectée, sinon (False, None)
    """
    frame_read = tello.get_frame_read()
    time.sleep(1)

    # Crée un dossier session horodaté
    session_name = time.strftime("session_%Y%m%d_%H%M%S")
    os.makedirs(f"log/{session_name}", exist_ok=True)

    detected = False
    detected_frame = None

    for i in range(12):
        frame = frame_read.frame
        frame = cv2.resize(frame, (640, 480))

        # Analyse
        bbox, label, conf = cv.detect_common_objects(frame, model='yolov3')
        frame_with_boxes = draw_bbox(frame.copy(), bbox, label, conf)

        # Sauvegarde l’image annotée
        fname = f"log/{session_name}/frame_{i*30}deg.jpg"
        cv2.imwrite(fname, frame_with_boxes)

        # Log texte
        if 'person' in label:
            detected = True
            detected_frame = frame
            with open(f"log/{session_name}/result.txt", "a") as f:
                f.write(f"[{i*30}°] Personne détectée\n")
            if verbose:
                print(f"[{i*30}°] Personne détectée")
            break
        else:
            with open(f"log/{session_name}/result.txt", "a") as f:
                f.write(f"[{i*30}°] Aucun humain détecté\n")
            if verbose:
                print(f"[{i*30}°] Aucun humain détecté")

        # Tourner
        tello.rotate_clockwise(30)
        time.sleep(2)

    return detected, detected_frame


def scan_for_human_yolov8(tello, verbose=True):
    """
    Fait tourner le drone de 360° par pas de 30°, analyse chaque vue avec YOLOv8n.
    Sauvegarde les frames annotées + logs.
    Retourne (True, frame) si une personne est détectée, sinon (False, None)
    """
    # Charger YOLOv8n en local
    model = YOLO("models/yolov8n.pt")  # chemin à adapter si besoin

    frame_read = tello.get_frame_read()
    time.sleep(1)

    # Dossier horodaté
    session_name = time.strftime("session_%Y%m%d_%H%M%S")
    os.makedirs(f"log/{session_name}", exist_ok=True)

    detected = False
    detected_frame = None

    for i in range(12):
        angle = i * 30
        frame = frame_read.frame
        frame = cv2.resize(frame, (640, 480))

        # Prédiction YOLOv8
        results = model.predict(source=frame, save=False, verbose=False)

        # Extraire les classes détectées
        for r in results:
            boxes = r.boxes
            classes = boxes.cls.cpu().numpy().astype(int)
            if 0 in classes:  # 0 = 'person'
                detected = True
                detected_frame = frame

                # Annoter et sauvegarder l'image
                r.save(filename=f"log/{session_name}/frame_{angle}deg.jpg")

                if verbose:
                    print(f"[{angle}°] 1;Human detected")
                with open(f"log/{session_name}/result.txt", "a") as f:
                    f.write(f"[{angle}°] 1;Human detected\n")
                break
            else:
                # Annoter et sauvegarder même sans détection
                r.save(filename=f"log/{session_name}/frame_{angle}deg.jpg")
                if verbose:
                    print(f"[{angle}°];0;No human detected")
                with open(f"log/{session_name}/result.txt", "a") as f:
                    f.write(f"[{angle}°];0;No human detected\n")

        if detected:
            break

        tello.rotate_clockwise(30)
        time.sleep(2)

    return detected, detected_frame
