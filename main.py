import cv2
import math
import time
from djitellopy import Tello
import os
import cvlib as cv
from cvlib.object_detection import draw_bbox
from ultralytics import YOLO
import numpy as np

os.makedirs("debug", exist_ok=True)

points = [(300, 200), (680, 390), (680, 560), (1150, 560), (205, 560)]
parcours = [1, 2, 3, 4, 3, 5, 3, 2, 1]

# Dictionnaire « QR-content ➜ position carte (cm) »
QR_POS = {
    "asf":  (300,   0),   # amer_salon_frigo
    "asr":  (0,   125),   # amer_salon_radiateur
    "asc":  (727, 450),   # amer_salon_couloir
    "ac1l": (205, 848),   # amer_chambre1_lit
    "ac2f": (1419, 764),  # amer_chambre2_fenetre
    "act":  (680, 608),   # amer_couloir_toilette
}

QR_SIZE_CM = 14                     # côté du QR imprimé
calib = np.load("tello_intrinsics_7x9.npz")     # ← chemin complet si besoin
# ≈ [[916,0,516],[0,916,367],[0,0,1]]
CAM_MTX = calib["K"].astype(np.float32)
DIST_COEF = calib["dist"].astype(np.float32)    # 5 coeffs

print("K  :", CAM_MTX, sep="\n")
print("dist:", DIST_COEF.ravel())
# Repère global des amers
QR_POS = {                  # x, y en cm
    "asf": (300,   0),
    "asr": (0, 125),
    "asc": (727, 450),
    "ac1l": (205, 848),
    "ac2f": (1419, 764),
    "act": (680, 608)
}

# Objet QR 3D : 4 points dans le plan z=0
objp = np.array([[0, 0, 0],
                 [QR_SIZE_CM, 0, 0],
                 [QR_SIZE_CM, QR_SIZE_CM, 0],
                 [0, QR_SIZE_CM, 0]], dtype=np.float32)

qr = cv2.QRCodeDetector()


def _pose_from_corners(corners):
    """corners: (4,2) px  →  rvec, tvec"""
    ret, rvec, tvec = cv2.solvePnP(objp, corners, CAM_MTX, DIST_COEF)
    return rvec, tvec  # tvec en cm (dans repère caméra)


def reposition_with_qr(tello, verbose=True):
    """
    Tourne 360°; si un QR est lu → calcule (x,y) drone ≈ position_amer + tvec_2D
    Retourne  (pos_dronexy, True) ou (None, False)
    """
    frame_read = tello.get_frame_read()
    time.sleep(1)

    for step in range(12):
        frame = frame_read.frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        data, pts, _ = qr.detectAndDecode(frame)
        print(f"Raw data='{data}'  Pts={type(pts)}")
        ok_save = cv2.imwrite("debug/frame_raw"+str(step)+".jpg", frame)

        if data in QR_POS and pts is not None:
            corners = np.squeeze(pts).astype(np.float32)  # (4,2)
            rvec, tvec = _pose_from_corners(corners)

            # tvec : translation caméra→QR dans repère caméra  (x→droite, y→bas, z→avant)
            tx, ty, tz = tvec.flatten()   # cm
            # Projection sur le sol : on ignore la hauteur (tz) et l’axe vert.
            # Par convention on garde (x_cam, z_cam) —> (x_local, y_local)
            dx = tx                       # droite(+)/gauche(−)
            dy = -tz                       # avant(+)/arrière(−)

            amer_x, amer_y = QR_POS[data]
            drone_x = amer_x + dx
            drone_y = amer_y - dy

            if verbose:
                print(f"✅ QR {data} — amer @({amer_x},{amer_y}) cm  → "
                      f"Drone estimé @({drone_x:.1f},{drone_y:.1f}) cm")

            return (drone_x, drone_y), True

        if verbose:
            print(f"[{step*30}°] aucun QR")
        tello.rotate_clockwise(30)
        time.sleep(2)

    if verbose:
        print("aucun QR trouvé")
    return None, False


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
        # frame = cv2.resize(frame, (640, 480))

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


def rotate_to_yaw(tello, target_yaw_deg, tol=2):
    """Tourne le drone vers target_yaw (0-360°) en choisissant le plus court chemin."""
    current = tello.get_yaw() % 360           # ex. 323°
    delta = (target_yaw_deg - current + 540) % 360 - 180   # dans [-180;180]

    if abs(delta) < tol:
        return                                # déjà aligné

    if delta > 0:
        tello.rotate_clockwise(int(round(delta)))
    else:
        tello.rotate_counter_clockwise(int(round(-delta)))

    time.sleep(2)                             # stabilisation


def move_forward_safe(tello, dist_cm):
    remaining = dist_cm
    while remaining > 0:
        step = min(500, remaining)
        tello.move_forward(int(round(step)))
        remaining -= step
        time.sleep(2)


def go_to_point(tello, current_pos, target_pos):
    """
    Déplace le drone du point courant vers target_pos.
    - Calcule la direction (en XY monde) et tourne le drone
      en tenant compte de son yaw réel.
    - Avance ensuite par tranches de 500 cm.
    """
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]

    distance = math.hypot(dx, dy)                # cm
    target_yaw = (math.degrees(math.atan2(dy, dx))) % 360

    # 1) Orientation
    rotate_to_yaw(tello, target_yaw)

    # 2) Translation
    move_forward_safe(tello, distance)

    return target_pos


def maintain_altitude(tello, target=90, deadband=15):
    alt = tello.get_height()          # cm
    diff = target - alt
    if abs(diff) > deadband:
        if diff > 0:
            tello.move_up(int(diff))
        else:
            tello.move_down(int(-diff))


# img = cv2.imread("amers/amer_salon_frigo.png")
# data, pts, _ = qr.detectAndDecode(img)
# print(data, pts)

# Créer une instance du drone
tello = Tello()

# Connexion au drone
tello.connect()

# tello.set_video_resolution(tello.RESOLUTION_720P)
# tello.set_video_bitrate(tello.BITRATE_5MBPS)

print(f"Battery: {tello.get_battery()}%")
tello.streamon()
# Décollage
tello.takeoff()
time.sleep(2)

maintain_altitude(tello)

ok = False
current_pos, ok = reposition_with_qr(tello)

if ok:
    for i in range(1, len(parcours)):
        print("🧍 Scan humain")
        scan_for_human_yolov8(tello, verbose=True)

        target_pos = points[parcours[i] - 1]

        print(f"\nÉtape {i}: déplacement vers {target_pos}")
        current_pos = go_to_point(tello, current_pos, target_pos)

        print("Repositionnement à l'aide des amers")
        pos, ok = reposition_with_qr(tello)
        if ok:
            # mets à jour ta position estimée
            current_pos = list(pos)

        maintain_altitude(tello)
        time.sleep(1)

    print("Trajet terminé avec succès")
else:
    print("echec d'initialisation, aucun amer")

try:
    tello.land()
except Exception as e:
    print(f"landing error: {e}")

# Fermeture flux
try:
    tello.streamoff()
except:
    pass

# Fin session
try:
    tello.end()
except:
    pass
