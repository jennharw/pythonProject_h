import os
from pathlib import Path


def path_to_project_root(root_folder_name: str = 'pythonProject'):
    path = os.getcwd()
    while not str(path).endswith(root_folder_name):
        path = Path(path).parent
    return str(path)
