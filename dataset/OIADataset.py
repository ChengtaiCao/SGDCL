""" 
    Implementation for BDDOIA Dataset 
"""

import torch
import json
import os
import yaml
import torchvision

from torch.utils.data import Dataset
from PIL import Image


class BDDOIA(Dataset):
    """ BDD-OIA Data"""
    # initialization
    def __init__(self, imageRoot, actionRoot, reasonRoot):
        super(BDDOIA, self).__init__()
        
        self.imageRoot = imageRoot
        self.actionRoot = actionRoot
        self.reasonRoot = reasonRoot

        # action json
        with open(actionRoot, "r") as f:
            _action_data = json.load(f)
        # reason json
        with open(reasonRoot, "r") as f:
            _reason_data = json.load(f)
        
        # sort image-id pairs and image-reasonLabel by image name
        _action_data["images"] = sorted(_action_data["images"], key=lambda item: item["file_name"])
        _reason_data = sorted(_reason_data, key=lambda item: item['file_name'])

        # get image names of images and labels
        # sort id-actionLabel pairs by image id
        self.imges_names, self.actions, self.reasons = [], [], []
        _action_annotations = sorted(_action_data['annotations'], key=lambda item: item["id"])
        for i, img_item in enumerate(_action_data['images']):
            img_id = img_item["id"]
            # fileter samples with [0, 0, 0, 0, 0]
            # id-actionLabel pairs are sorted by image id
            if len(_action_annotations[img_id]['category']) == 4  or \
                   _action_annotations[img_id]['category'][4] == 0:
                file_name = os.path.join(self.imageRoot, img_item['file_name'])
                if os.path.isfile(file_name):
                    self.imges_names.append(file_name)
                    self.actions.append(_action_annotations[img_id]['category'][:4])
                    assert len(_reason_data[i]['reason']) == 21
                    self.reasons.append(_reason_data[i]['reason'])

        # get size of data
        self.count = len(self.imges_names)


    # size of data
    def __len__(self):
        return self.count


    # get item
    def __getitem__(self, i):
        # convert action_list and reason_list to tensors
        actions = torch.FloatTensor(self.actions[i])
        reasons = torch.FloatTensor(self.reasons[i])

        img_name = self.imges_names[i]
        raw_img = Image.open(img_name)
        img = self.pre_process(raw_img)  # torch.Size([3, 720, 1280])

        return img, [actions, reasons], img_name


    # pre-process img
    def pre_process(self, raw_img):
        color_jitter = torchvision.transforms.ColorJitter(
                brightness=0.0,
                contrast=0.0,
                saturation=0.0,
                hue=0.0,
            )
        img = color_jitter(raw_img)

        img = torchvision.transforms.functional.to_tensor(img)
        img = img[[2, 1, 0]] * 255
        img = torchvision.transforms.functional.normalize(
                img,
                mean=[102.9801, 115.9465, 122.7717],
                std=[1., 1., 1.]
            )
        return img



if __name__ == "__main__":
    # config
    config_file = "./bddoia_config.yaml"
    with open(f"{config_file}", 'r') as f:
        CONFIG = yaml.safe_load(f)

    bddoia_data = CONFIG["data"]["bddoia_data"]
    action_val = CONFIG["data"]["val_action"]
    reason_val = CONFIG["data"]["val_reason"]
    val_data = BDDOIA(bddoia_data, action_val, reason_val)
    print(len(val_data))
