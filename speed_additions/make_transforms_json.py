import json
import os
import random
import sys
import argparse
import numpy as np
from transforms import quaternion2rotation

def create_transf_matrix(q, t):
    last_line = np.zeros((1,4))
    last_line[0,3] = 1.0 # Last line of the 4x4 matrix

    Rinv = quaternion2rotation(q).T
    T = -Rinv@t
    Rx180 = np.array([[1.0, 0.0, 0.0],[0.0, -1.0, 0.0],[0.0, 0.0, -1.0]])
    R = Rinv @ Rx180
    M = np.concatenate((R,T), axis=1)
    M = np.concatenate((M,last_line), axis=0)

    return M.tolist()

parser = argparse.ArgumentParser()
parser.add_argument('-data','--dataset_locations',type=str,nargs='+',help='Provide a list of dataset locations to be included in transforms.json')
parser.add_argument('store_location', type=str, help='Provide the location where transforms.json should be stored')
parser.add_argument('-m','--method', type=str, help='What action will the dataset be used for \nDEFAULT: train', default='train', choices=['train','test','val'])
parser.add_argument('-split','--train_val_test_split', type=float,nargs='+',help='Declare a list with the split of your dataset (every value between 0 and 1) \nDEFAULT: 1 0 0', default=[1,0,0])
parser.add_argument('-lsf','--load_split_file', help='Use an existing split file, provided in --split_file_location', action='store_true')
parser.add_argument('-sfl','--split_file_location', type=str, help='The location of the split file to be loaded', default=None)
args = parser.parse_args()
assert len(args.train_val_test_split) in [3,0], 'Provide 3 numbers in the split: train, val and test'

print('-------------------------')
if args.load_split_file:
    print(f'Storing a transforms.json file with images already present in the splitfile {args.split_file_location}')
    print(f'The transforms.json file is stored at {args.store_location}. This is the location you need to specify under --data when training in nerfstudio!')
else:
    print(f'Storing a transforms.json file with images from the following datasets: {args.dataset_locations}')
    print(f'The transforms.json file and splitfile are stored at {args.store_location}. This is the location you need to specify under --data when training in nerfstudio!')
    print(f'The split of the data in the splitfile is: {args.train_val_test_split[0]*100}% for training, {args.train_val_test_split[1]*100}% for validation and {args.train_val_test_split[2]*100}% for testing')
print(f'In transforms.json is the metadata necessary for {args.method}')
print('-------------------------')

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

if args.load_split_file:
    assert args.split_file_location is not None, 'Provide a location of the split file!'
    with open(args.split_file_location, 'r', encoding='utf-8') as f:
        split_file = json.load(f)
    frames = split_file[args.method]
else:
    frames = []
    for dataset_location in args.dataset_locations:
        json_files = filter(lambda input: '.json' in input,os.listdir(dataset_location))
        metadata = []
        for filename in json_files:
            with open(dataset_location+'/'+filename,'r', encoding='utf-8') as f:
                metadata.extend(json.load(f))

        for img_data in metadata:
            img_filename = img_data['filename']
            img_filename = "images/" + img_filename
            t = np.array(img_data["r_Vo2To_vbs_true"], dtype=np.float32).reshape(3,1)
            if np.abs(t[2,0])<thresh_dist:
                frame = {}
                frame["file_path"] = os.path.relpath(dataset_location,start=args.store_location)+'/'+img_filename
                frame["transform_matrix"] = create_transf_matrix(np.array(img_data["q_vbs2tango_true"],  dtype=np.float32),t)
                frames.append(frame)

    random.shuffle(frames)
    assert np.sum(args.train_val_test_split)==1, 'The sum of the different splits in train_val_test_split should be 1!'
    train_length = int(np.floor(args.train_val_test_split[0]*len(frames)))
    val_length = int(np.floor(args.train_val_test_split[1]*len(frames)))
    splitfile = {'train': frames[:train_length], 'val': frames[train_length:train_length+val_length], 'test': frames[train_length+val_length:]}
    with open(args.store_location+f'/splitfile{int(args.train_val_test_split[0]*100):02d}{int(args.train_val_test_split[1]*100):02d}{int(args.train_val_test_split[2]*100):02d}.json','w', encoding='utf-8') as f:
        json.dump(splitfile, f, indent=2)
    frames = splitfile[args.method]

data["frames"] = frames
with open(args.store_location+'/transforms.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print('\U0001F389 \U0001F389 DONE \U0001F389 \U0001F389')
