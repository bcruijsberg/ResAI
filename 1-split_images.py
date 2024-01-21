import math
from pathlib import Path
import random
import shutil

#####
# When images are labeled in Roboflow, they can be exported with a certain split for training, validation, and testing.
# In order to have control over this split a RoboFlow "trainingset" is created with a split of 150-1-1.
# This trainingset is then exported to the project folder
# Once exported the 150 'train'images are then copied into the 'data' folder of the project. The remaining 2 images are
# ignored.
# 6 datasets are created with different splits of the 150 images.
# The splits are 100-25-25, 80-20-20, 60-15-15, 40-10-10, 20-5-5
# For each split 10 datasets are created with different random splits of the 150 images.
# The new datasets are created in the 'shuffled' folder.
# The datasets are named SL_100_25_25, SL_80_20_20, SL_60_15_15, SL_40_10_10, SL_20_5_5
# The subsets are named 0,1,2,3,4,5,6,7,8,9
# Each subset contains a train, valid and test folder with images and labels.
# The data.yaml file is copied to each subset.
# The content of the data.yaml file is:
#   train: ./train/images
#   val: ./valid/images
#   test: ./test/images
#
#   nc: 1
#   names: ['lemon']
#
# The subsets are then used to train the model.




# Function to copy images and labels to respective directories
def copy_files(file_list, destination_img, destination_lbl):
    for image_path in file_list:
        shutil.copy(image_path, destination_img)
        label_path = labels_dir / image_path.with_suffix('.txt').name
        shutil.copy(label_path, destination_lbl)


# Define your dataset directory and the directories for the splits
project_dir = Path('C:/Users/bcrui/Documents/Yolov8/datasets/LemonsSize150/')
images_dir = project_dir / 'data3/images'
labels_dir = project_dir / 'data3/labels'
yaml_file = project_dir / 'data.yaml'

# Get a list of all image files in the dataset
all_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))


for n in range(0, 10):
    for i in range(1, 6):

        # Shuffle the list for random split
        random.shuffle(all_images)

        # Calculate the split percentages based on the number of images wanted for the trainingset (multiple of 20)
        train_per = i*20/len(all_images)
        val_per = (train_per / .8 - train_per)

        # Calculate the actual number of images for each split
        train_split = int(train_per * len(all_images))
        val_split = train_split + int(math.ceil(val_per * len(all_images)))

        # folder and subfolder names
        trainingset = f"SL_{train_split}_{val_split - train_split}_{len(all_images) - val_split}"
        subfolder = f"{n}"

        session = project_dir / 'shuffled_bobz_7' / trainingset / subfolder
        train_dir_img = session / 'train' / 'images'
        val_dir_img = session / 'valid' / 'images'
        test_dir_img = session / 'test' / 'images'

        train_dir_lbl = session / 'train' / 'labels'
        val_dir_lbl = session / 'valid' / 'labels'
        test_dir_lbl = session / 'test' / 'labels'

        # Create directories for the splits if they don't exist
        train_dir_img.mkdir(parents=True, exist_ok=True)
        val_dir_img.mkdir(parents=True, exist_ok=True)
        test_dir_img.mkdir(parents=True, exist_ok=True)
        train_dir_lbl.mkdir(parents=True, exist_ok=True)
        val_dir_lbl.mkdir(parents=True, exist_ok=True)
        test_dir_lbl.mkdir(parents=True, exist_ok=True)

        # Split the dataset
        train_images = all_images[:train_split]
        val_images = all_images[train_split:val_split]
        test_images = all_images[val_split:]


        # Copy files to their respective directories
        copy_files(train_images, train_dir_img, train_dir_lbl)
        copy_files(val_images, val_dir_img, val_dir_lbl)
        copy_files(test_images, test_dir_img, test_dir_lbl)
        shutil.copy(yaml_file, session / 'data.yaml')


