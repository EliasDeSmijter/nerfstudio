import json
import os
import random
import tyro
from typing import Literal, Tuple
import numpy as np
from .transforms import quaternion2rotation

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

def make_transforms_json(store_location: str, dataset_locations: Tuple[str, ...], method: Literal['train','val','test']='train', split: Tuple[float, float, float]=(1,0,0), load_split_file: bool=False, split_file_loc: str=None) -> None:
    """Generate transforms.json file for a given dataset.
    Some parts are hardcoded for speedplusv2!

    Args:
        store_location: Location where transforms.json will be stored, along with the generated split file if no existing split file is used.
        dataset_locations: List of locations of datasets to be included.
        method: For what will transforms.json be used?
        split: List with 3 floats specifying how the dataset should be split into train, validation and test set.
        load_split_file: If used: use the dataset already specified in an earlier generated split file.
        split_file_loc: Location of the split file to be used.
    """
    assert method in ['train', 'val', 'test'], "The method should be train, val or test !"
    assert len(split)==3, 'Provide 3 numbers in the split: train, val and test'

    print('-------------------------')
    if load_split_file:
        print(f'Storing a transforms.json file with images already present in the splitfile {split_file_loc}')
        print(f'The transforms.json file is stored at {store_location}. This is the location you need to specify under --data when training in nerfstudio!')
    else:
        print(f'Storing a transforms.json file with images from the following datasets: {dataset_locations}')
        print(f'The transforms.json file and splitfile are stored at {store_location}. This is the location you need to specify under --data when training in nerfstudio!')
        print(f'The split of the data in the splitfile is: {split[0]*100}% for training, {split[1]*100}% for validation and {split[2]*100}% for testing')
    print(f'In transforms.json is the metadata necessary for {method}')
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

    if load_split_file:
        assert split_file_loc is not None, 'Provide a location of the split file!'
        with open(split_file_loc, 'r', encoding='utf-8') as f:
            split_file = json.load(f)
        frames = split_file[method]
    else:
        frames = []
        for dataset_location in dataset_locations:
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
                    frame["file_path"] = os.path.relpath(dataset_location,start=store_location)+'/'+img_filename
                    frame["transform_matrix"] = create_transf_matrix(np.array(img_data["q_vbs2tango_true"],  dtype=np.float32),t)
                    frames.append(frame)

        random.shuffle(frames)
        assert np.sum(split)==1, 'The sum of the different splits in split should be 1!'
        train_length = int(np.floor(split[0]*len(frames)))
        val_length = int(np.floor(split[1]*len(frames)))
        splitfile = {'train': frames[:train_length], 'val': frames[train_length:train_length+val_length], 'test': frames[train_length+val_length:]}
        with open(store_location+f'/splitfile{int(split[0]*100):02d}{int(split[1]*100):02d}{int(split[2]*100):02d}.json','w', encoding='utf-8') as f:
            json.dump(splitfile, f, indent=2)
        frames = splitfile[method]

    data["frames"] = frames
    with open(store_location+'/transforms.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print('\U0001F389 \U0001F389 DONE \U0001F389 \U0001F389')

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(make_transforms_json)

if __name__ == "__main__":
    entrypoint()
