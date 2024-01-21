from pathlib import Path
import pandas as pd
from pprint import pprint as pp



if __name__ == '__main__':

    # load the csv file with the results of the training of validationfrom the runs/train or runs/val
    # folder of the dataset and merge them into one dataframe adding a column with the name of the dataset/folder/subfolder

    data_set = pd.DataFrame()


    project_dir = Path(r"C:/Users/bcrui/Documents/Yolov8/datasets/LemonsSize150/")
    datasets = project_dir / 'shuffled_bobz_9'

    # retreive all folders in the shuffled folder
    folders = [x for x in datasets.iterdir() if x.is_dir()]

    for folder in folders:
        # retreive all subfolders in the shuffled folder
        subfolders = [x for x in folder.iterdir() if x.is_dir()]
        # loop through all subfolders
        for subfolder in subfolders:
            df = pd.read_csv(subfolder / "runs/val/results_validation10.csv")
            df['dataset'] = (f"{folder.name}")
            df['run'] = (f"{subfolder.name}")

            #print(df.head(), df.shape)
            data_set = pd.concat([data_set, df], ignore_index=True)
            print(data_set.head(), data_set.shape)
            
    # save to csf in teh project folder
    data_set.to_csv(datasets / "resultso_val10.csv", index=False)

