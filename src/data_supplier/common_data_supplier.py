import os
import joblib

def file_path_from_ncode(ncode, dir_path):
    return os.path.join(dir_path, ncode + '.txt')

def load_from_ncode(ncode, dir_path):
    file_path = file_path_from_ncode(ncode, dir_path)
    with open(file_path , 'rb') as f:
        data = joblib.load(f)
    return data
