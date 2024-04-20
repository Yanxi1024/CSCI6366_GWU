import os

from pandas import DataFrame

"""
A demo for generator DIF(datasets' information file) according to "Folder Classification" format.
"""


def get_class_list(sound_dir):
    return os.listdir(sound_dir)

def generator(csv_path, sound_dir):
    """
    data items:filename,filepath,label
    """
    class_list = get_class_list(sound_dir)
    data = {'filename': [], 'filepath': [], 'label': []}
    with open(os.path.join(os.path.dirname(csv_path), 'classes.txt'), 'w', encoding='utf-8') as class_file:
        for idx, class_name in enumerate(class_list):
            class_file.write(f"{idx} {class_name}\n")
            class_path = os.path.join(sound_dir, class_name)
            for filename in os.listdir(class_path):
                filepath = os.path.join(class_path, filename)
                data['filename'].append(filename)
                data['filepath'].append(filepath)
                data['label'].append(idx)
    dataframe = DataFrame(data=data)
    dataframe.to_csv(csv_path, encoding='utf-8')

if __name__ == '__main__':
    csv_path = r'./refer.csv'
    sound_dir = r'./Sound'
    generator(csv_path, sound_dir)