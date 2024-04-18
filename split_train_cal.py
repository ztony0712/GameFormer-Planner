import os
import shutil
from random import shuffle

def split_data_into_train_and_valid(src_folder, train_folder, valid_folder, split_ratio=0.8):
    # Create train and valid folders if they don't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)

    # List all the files in the src_folder
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    shuffle(all_files)

    # Calculate the split index
    split_index = int(len(all_files) * split_ratio)

    # Split files into train and valid
    train_files = all_files[:split_index]
    valid_files = all_files[split_index:]

    # Move each train file into train_folder
    for file_name in train_files:
        src = os.path.join(src_folder, file_name)
        dst = os.path.join(train_folder, file_name)
        shutil.move(src, dst)

    # Move each valid file into valid_folder
    for file_name in valid_files:
        src = os.path.join(src_folder, file_name)
        dst = os.path.join(valid_folder, file_name)
        shutil.move(src, dst)

if __name__ == "__main__":
    src_folder = os.path.expanduser("/media/nuplan/data2/Datasets/gameformer_processed")
    train_folder = os.path.expanduser("/media/nuplan/data2/Datasets/gameformer-v1.0/train")
    valid_folder = os.path.expanduser("/media/nuplan/data2/Datasets/gameformer-v1.0/test")

    split_data_into_train_and_valid(src_folder, train_folder, valid_folder, split_ratio=0.9)
