from roboflow import Roboflow
import os
import shutil

def download_and_move_dataset(version_number=14, output_dir="data/football-players-v14"):
    rf = Roboflow(api_key="0G8tn3HwWqn76g48X75u")
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(version_number)
    
    print(f"Downloading version {version_number}...")
    dataset = version.download("yolov8")  #  to current working directory
    downloaded_dir = dataset.location   

    print(f"Downloaded to: {downloaded_dir}")

    # Move   
    abs_target_dir = os.path.join(os.getcwd(), output_dir)
    
    if os.path.exists(abs_target_dir):
        print(f"Deleting existing directory: {abs_target_dir}")
        shutil.rmtree(abs_target_dir)
    
    print(f"Moving dataset to: {abs_target_dir}")
    shutil.move(downloaded_dir, abs_target_dir)
    print("Done.")

if __name__ == "__main__":
    download_and_move_dataset()
