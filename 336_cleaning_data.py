import os
import shutil

def delete_unwanted_folders(root_dir, retain_folders):
    for foldername in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, foldername)
        if os.path.isdir(folder_path):
            for subfoldername in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfoldername)
                if os.path.isdir(subfolder_path) and subfoldername not in retain_folders:
                    print(f"Deleting folder: {subfolder_path}")
                    shutil.rmtree(subfolder_path)

if __name__ == "__main__":
    root_directory = './data'  # Change this to your directory
    folders_to_retain = ['label masks modify', 'mask binary without border', 'tissue images']

    delete_unwanted_folders(root_directory, folders_to_retain)
