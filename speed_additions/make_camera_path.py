import tyro
import numpy as np
import json
from .transforms import random_quaternion,quaternion2rotation

def generate_random_camera_path(nb_positions: int=100, filename: str='camera_path.json') -> None:
    """Function to generate a random camera path to be used when rendering
    
    Args:
        nb_positions: Number of positions in the camera path
        filename: path + name of file to be saved. Default: save in current working directory
    """
    data = {}
    data["camera_type"] = "perspective"
    data["render_height"] = 1200
    data["render_width"] = 1920

    last_line = np.zeros((1,4))
    last_line[0,3] = 1.0

    dist_min = 2.5
    dist_max = 10.0

    dist_min /= 3.0 # account for NeRFStudio scaling
    dist_max /= 3.0 # account for NeRFStudio scaling

    camera_path = []
    for _ in range(nb_positions):
        q = random_quaternion()
        d = np.random.random() * (dist_max - dist_min) + dist_min
        dx = ((np.random.random()*2.0)-1.0) * d * 0.15
        dy = ((np.random.random()*2.0)-1.0) * d * 0.15
        t = np.array([[dx],[dy],[d]])

        Rinv = quaternion2rotation(q).T
        T = -Rinv@t
        Rx180 = np.array([[1.0, 0.0, 0.0],[0.0, -1.0, 0.0],[0.0, 0.0, -1.0]])
        R = Rinv @ Rx180
        M = np.concatenate((R,T), axis=1)
        M = np.concatenate((M,last_line), axis=0).reshape(16,)

        frame = {}
        frame["camera_to_world"] = M.tolist()
        frame["fov"] = 22.595
        frame["aspect"] = 1.6
        camera_path.append(frame)

    data["camera_path"] = camera_path
    data["fps"] = 1
    data["seconds"] = nb_positions
    data["smoothness_value"] = 0.0
    data["is_cycle"] = False
    data["crop"] = None

    with open(filename, 'w', encoding='utf-8') as f: # Camera Path for NeRFStudio
        json.dump(data, f, indent=2)


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(generate_random_camera_path)
    return

if __name__ == "__main__":
    entrypoint()
