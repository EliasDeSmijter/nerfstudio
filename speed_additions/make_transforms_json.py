import json
import numpy as np
from transforms import *
import sys


def make_file(dataset_location: str, type='train'):
    thresh_dist = 5.0 # Interspacecraft distance max
    select_proba = 1.0 # Probability of selecting a valid frame from SPEED to 

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

    with open(dataset_location+'/'+type+'.json','r') as f:
        metadata = json.load(f)

    last_line = np.zeros((1,4))
    last_line[0,3] = 1.0 # Last line of the 4x4 matrix

    frames = []
    for index in range(len(metadata)):
        img_data = metadata[index]
        img_filename = img_data['filename']
        if index==0:
            print(img_filename)
        img_filename = "images/" + img_filename
        
        if "q_vbs2tango" in img_data.keys():
            q = np.array(img_data["q_vbs2tango"],  dtype=np.float32)
        elif "q_vbs2tango_true" in img_data.keys():
            q = np.array(img_data["q_vbs2tango_true"],  dtype=np.float32)
        else:
            raise KeyError("q_vbs2tango key is not present?")      
        t = np.array(img_data["r_Vo2To_vbs_true"], dtype=np.float32).reshape(3,1)
        
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
    with open(dataset_location+'/transforms.json', 'w') as f:
        json.dump(data, f, indent=2)

    return

if __name__=='__main__':
    if len(sys.argv)==2:
        print(f'Storing transforms file for train at {sys.argv[1]}')
        make_file(sys.argv[1])
    elif len(sys.argv)>2:
        print(f'Storing transforms file for {sys.argv[2]} at {sys.argv[1]}')
        make_file(sys.argv[1], type=sys.argv[2])
    else:
        print('Add path of dataset!')