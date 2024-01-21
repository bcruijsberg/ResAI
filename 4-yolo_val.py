import ultralytics
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import os
import torch
import gc
####
# This script is used to validate the models on the validation set of 75 images
####



if __name__ == '__main__':

    project_dir = Path(r"C:/Users/bcrui/Documents/Yolov8/datasets/LemonsSize150/shuffled_bobz_9/")
    folders = [x for x in project_dir.iterdir() if x.is_dir()]
    df =pd.DataFrame()

    for folder in folders:

        # retreive all folders in the shuffled folder
        subfolders = [y for y in folder.iterdir() if y.is_dir()]

        #for folder in folders:
            # retreive all subfolders in the shuffled folder
            #  subfolders = [x for x in folder.iterdir() if x.is_dir()]
            # loop through all subfolders
        for subfolder in subfolders:
            print(f"Starting in subfolder: {subfolder}\n")
            
            #retrieve the trained model
            classifier_file = project_dir / folder / subfolder / 'runs' / 'train' / 'weights' / 'best.pt'
            model = YOLO(str(classifier_file))
            
            #validate the model
            dataset = f"{project_dir}/{folder.name}/{subfolder.name}/data.yaml"
            project = f"{project_dir}/{folder.name}/{subfolder.name}/runs"
            model.val(data=dataset, plots=True, workers=1, project=project, conf=0.9, device='cuda')  # It'll automatically evaluate the data you trained.
            new_data = pd.DataFrame([model.metrics.results_dict])
            new_data['dataset'] = (f"{folder.name}")
            new_data['run'] = (f"{subfolder.name}")
            df = pd.concat([df, new_data], ignore_index=True)
            
            # Create the new directory
            #new_directory = project_dir / folder / subfolder / 'runs' / 'valid'
            #os.makedirs(new_directory, exist_ok=True)

            #write results
            new_data.to_csv(project_dir / folder / subfolder / 'runs' / 'val' / "results_validation10.csv", index=False)
            print(f"Saved:\n {new_data.shape} {new_data.head()}")
            gc.collect()
            torch.cuda.empty_cache()
