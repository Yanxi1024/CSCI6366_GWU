import argparse

import torch
from torchvision.transforms import ToTensor
from CatSoundClassification.DataSets.txt2classlist import trans
import numpy as np


def get_arg():
    parser = argparse.ArgumentParser(description='Audio classification parameter configuration(train)')
    parser.add_argument(
        '-t',
        type=str,
        default='CatSoundClassification',
        help="the theme's name of your task"
    )
    parser.add_argument(
        '-wp',
        type=str,
        default=r'checkpoints/best_f1.pth',
        help="the checkpoint applied to predict"
    )
    parser.add_argument(
        '-fp',
        type=str,
        default=r'./test(label0).npy',
        help="the audio feature file' path"
    )
    parser.add_argument(
        '-classes',
        type=list,
        default=trans(r'/Users/ruiyangchen/Desktop/Course/6366/CSCI6366_GWU/CatSoundClassification/Data/scatter/classes.txt'),
        help="classes list"
    )
    return parser.parse_args()

args = get_arg()
# ----------------------------------------------------------------------------------------------------------------------
# Load model
if torch.cuda.is_available():
    print("Predict on cuda and there are/is {} gpus/gpu all.".format(torch.cuda.device_count()))
    print("Device name:{}\nCurrent device index:{}.".format(torch.cuda.get_device_name(), torch.cuda.current_device()))
else:
    print("Predict on cpu.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Load weight from the path:{}.".format(args.wp))
model = torch.load(args.wp)
model = model.to(device)
inputs = np.load(args.fp)
inputs = ToTensor()(inputs).permute(1, 2, 0).unsqueeze(0).to(device)
# ----------------------------------------------------------------------------------------------------------------------
# Forward propagation for prediction
output = model(inputs)  # shape: ( N * cls_n )
output_ = output.clone().detach().cpu()
_, pred = torch.max(output_, 1)  # Output the index of the maximum probability for each row (sample)
print("Prediction:", args.classes[int(pred)])








