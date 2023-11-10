import json
import numpy as np
from transforms import *

thresh_dist = 5.0 # Interspacecraft distance max

data = {}
### camera parameters
data["fl_x"] = 2988.5795163815555
data["fl_y"] = 2988.3401159176124
data["k1"] = -0.22383016606510672
data["k2"] = 0.51409797089106379
data["p1"] = -0.00066499611998340662
data["p2"] = -0.00021404771667484594
data["p3"] = -0.13124227429077406
data["cx"] = 960.0
data["cy"] = 600.0
data["w"] = 1920
data["h"] = 1200

# iNGP params
data["aabb_scale"] = 0.5
data["n_extra_learnable_dims"] = 16

speed_folder = '../datasets/speed'
with open(speed_folder+'/train.json','r') as f:
    metadata = json.load(f)

last_line = np.zeros((1,4))
last_line[0,3] = 1.0 # Last line of the 4x4 matrix

frames = []
for index in range(len(metadata)):
    img_filename = metadata[index]['filename']
    if index==0:
        print(img_filename)
    img_filename = "train/" + img_filename
    q = np.array(metadata[index]["q_vbs2tango"],  dtype=np.float32)
    t = np.array(metadata[index]["r_Vo2To_vbs_true"], dtype=np.float32).reshape(3,1)
    
    Rinv = quaternion2rotation(q).T
    T = -Rinv@t
    Rx180 = np.array([[1.0, 0.0, 0.0],[0.0, -1.0, 0.0],[0.0, 0.0, -1.0]])
    R = Rinv @ Rx180
    M = np.concatenate((R,T), axis=1)
    M = np.concatenate((M,last_line), axis=0)
    
    if(np.abs(t[2,0])<thresh_dist and np.random.rand()<select_proba):
        frame = {}
        frame["file_path"] = img_filename
        frame["transform_matrix"] = M.tolist()
        frames.append(frame)
        
data["frames"] = frames
with open('transforms.json', 'w') as f:
    json.dump(data, f, indent=2)