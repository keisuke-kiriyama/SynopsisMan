from os import path, pardir

FILE_PATH = path.abspath(__file__)
UTIL_PATH = path.dirname(FILE_PATH)
SRC_DIR_PATH = path.abspath(path.join(UTIL_PATH, pardir))
PROJECT_ROOT = path.abspath(path.join(SRC_DIR_PATH, pardir))
DATA_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'data'))
MODEL_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'model'))

def check():
    print("FILE_PATH: ", FILE_PATH)
    print("UTIL_PATH: ", UTIL_PATH)
    print("SRC_DIR_PATH: ", SRC_DIR_PATH)
    print("PROJECT_ROOT: ", PROJECT_ROOT)
    print("DATA_DIR_PATH: ", DATA_DIR_PATH)
    print("MODEL_DIR_PATH: ", MODEL_DIR_PATH)
