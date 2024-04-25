import argparse

import os
import torch
from torchvision.transforms import ToTensor
from CSCI6366_GWU.CatSoundClassification.DataSets.txt2classlist import trans
from CSCI6366_GWU.CatSoundClassification.Tools.data_npy import generator
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
        default=r'../WorkDir/exp-CatSoundClassification_2024_4_25_0_52/checkpoints/best_f1.pth',
        help="the checkpoint applied to predict"
    )
    parser.add_argument(
        '-dp',
        type=str,
        default=r'../Data/folder/SoundTest',
        help="the audio feature file' path"
    )
    parser.add_argument(
        '-classes',
        type=list,
        default=trans(r'../Data/scatter/classes.txt'),
        help="classes list"
    )
    return parser.parse_args()

if __name__ == '__main__':
    # process sound test
    csv_path = r'../TestSet/refer.csv'
    sound_dir = r'../Data/folder/SoundTest'
    npy_dir = r'../TestSet/npy_data'
    target_duration = 5.0  # Set the target duration in seconds
    target_sr = 44100  # Set the target sample rate to 44.1kHz
    npy_data = generator(csv_path, sound_dir, npy_dir, target_duration)


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
    #
    # inputs = np.load(args.fp)
    # inputs = ToTensor()(inputs).permute(1, 2, 0).unsqueeze(0).to(device)
    # # ----------------------------------------------------------------------------------------------------------------------
    # # Forward propagation for prediction
    # output = model(inputs)  # shape: ( N * cls_n )
    # output_ = output.clone().detach().cpu()
    # _, pred = torch.max(output_, 1)  # Output the index of the maximum probability for each row (sample)
    # print("Prediction:", args.classes[int(pred)])
    for class_name in os.listdir(args.dp):
        class_dir = os.path.join(args.dp, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                inputs = np.load(file_path)
                inputs = ToTensor()(inputs).permute(1, 2, 0).unsqueeze(0).to(device)

                output = model(inputs)
                _, pred = torch.max(output.detach().cpu(), 1)

                predicted_class = args.classes[int(pred)]
                print(f'File: {file}, True Class: {class_name}, Predicted Class: {predicted_class}')
