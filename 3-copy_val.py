import math
from pathlib import Path
import random
import shutil

####
# This script is used to first remove the initial validation sets, which focused on only one scenario
# Next the validation set with 75 images of each scenario (25 of each) was copied to each folder, ready for validation
####

split_folders =['SL_100_25_25','SL_80_20_50','SL_60_15_75','SL_40_10_100','SL_20_5_125']
for split in split_folders:
    for n in range(0, 100):
        
        #source_valid= project_dir / 'val75'
        destinaton_valid = Path(f'C:/Users/bcrui/Documents/Yolov8/datasets/LemonsSize150/shuffled_bobz_9/{split}/{n}')

        print(destinaton_valid / 'valid')
        
        # Remove the folder and all its contents
        shutil.rmtree(destinaton_valid / 'valid')
    
    
    for n in range(0, 100):

        # set source and destination directory
        source_valid = Path(f'C:/Users/bcrui/Documents/Yolov8/datasets/LemonsSize150/val75-3/valid')
        
        #source_valid= project_dir / 'val75'
        destinaton_valid = Path(f'C:/Users/bcrui/Documents/Yolov8/datasets/LemonsSize150/shuffled_bobz_9/{split}/{n}')
        
        #copy new folder
        shutil.copytree(source_valid, destinaton_valid / 'valid')
        