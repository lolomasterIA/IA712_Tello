from djitellopy import Tello
import time
import detecthuman as u


# Créer une instance du drone
tello = Tello()

# Connexion au drone
tello.connect()

print(f"Battery: {tello.get_battery()}%")
tello.streamon()
# Décollage
tello.takeoff()
time.sleep(1)

is_person, frame = u.scan_for_human_yolov8(tello, False)
print(is_person)

# Atterrissage
# Atterrissage sécurisé
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
