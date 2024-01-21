import ultralytics
from ultralytics import YOLO
from pathlib import Path
import torch
import gc

####
# This script is used to train the model on the different subsets of the dataset.
# The subsets are created with the split_images.py script.
# The subsets are named SL_100_25_25, SL_80_20_20, SL_60_15_15, SL_40_10_10, SL_20_5_5
# The subsets are then used to train the model.
####

#### NOTE ####
# There seems to be a memory leak / bug in the YOLO class. If you run this script multiple times in a row,
# the memory usage increases with each run. This is not the case when you run the script from the command line.
####


if __name__ == '__main__':

    #split_folders =['SL_100_25_25','SL_80_20_50','SL_60_15_75','SL_40_10_100','SL_20_5_125']
    split_folders =['temp']
    for split in split_folders:
        project_dir = Path(r"C:/Users/bcrui/Documents/Yolov8/datasets/LemonsSize150/")
        datasets = project_dir / f'shuffled_bobz_9/{split}'

        # retreive all folders in the shuffled folder
        subfolders = [x for x in datasets.iterdir() if x.is_dir()]

        #for folder in folders:
            # retreive all subfolders in the shuffled folder
            #  subfolders = [x for x in folder.iterdir() if x.is_dir()]
            # loop through all subfolders
        for subfolder in subfolders:
            print(f"Starting in subfolder: { subfolder}\n")

            # train the model
            model = YOLO("yolov8s.pt")
            model.train(data=subfolder / "data.yaml", epochs=25, imgsz=640, plots=True, project=subfolder / "runs",workers=1)
            del model
            gc.collect()
            torch.cuda.empty_cache()
