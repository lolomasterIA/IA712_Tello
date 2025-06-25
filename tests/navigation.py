import yaml
import math
# suppose que tu as une classe pour envoyer les commandes au Tello
from drone_controller import DroneController


class Navigator:
    def __init__(self, map_path):
        with open(map_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.amers = self.config['amers']
        self.trajectory = self.config['trajectory']
        self.current_position = self.amers[self.trajectory[0]]['pos']
        self.dc = DroneController()

    def compute_relative_move(self, target_pos):
        x0, y0 = self.current_position
        x1, y1 = target_pos
        dx = x1 - x0
        dy = y1 - y0
        distance = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx))
        return distance, angle, target_pos

    def move_to(self, name):
        target_pos = self.amers[name]['pos']
        distance, angle, _ = self.compute_relative_move(target_pos)

        self.dc.rotate_to_angle(angle)
        self.dc.forward(distance)

        print(f"Arrivé à {name} : position théorique {target_pos}")
        self.current_position = target_pos

    def execute_trajectory(self):
        for name in self.trajectory[1:]:
            self.move_to(name)
            self.dc.scan_amer(name)  # Optionnel : recaler via image captée
            self.dc.hover()

# Utilisation
# navigator = Navigator("map.yaml")
# navigator.execute_trajectory()
