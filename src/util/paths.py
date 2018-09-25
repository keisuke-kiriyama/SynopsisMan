from os import path, pardir

FILE_PATH = path.abspath(__file__)
UTIL_PATH = path.dirname(FILE_PATH)
SRC_DIR_PATH = path.abspath(path.join(UTIL_PATH, pardir))
PROJECT_ROOT = path.abspath(path.join(SRC_DIR_PATH, pardir))
DATA_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'data'))
ORIGIN_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'origin'))
ORIGIN_CONTENTS_DATA_DIR_PATH = path.abspath(path.join(ORIGIN_DATA_DIR_PATH, 'contents'))
ORIGIN_META_DATA_DIR_PATH = path.abspath(path.join(ORIGIN_DATA_DIR_PATH, 'meta'))
PREPROCESSED_DATA_DIR_PATH = path.abspath(path.join(DATA_DIR_PATH, 'preprocessed'))
PREPROCESSED_CONTENTS_DATA_DIR_PATH = path.abspath(path.join(PREPROCESSED_DATA_DIR_PATH, 'contents'))
PREPROCESSED_META_DATA_DIR_PATH = path.abspath(path.join(PREPROCESSED_DATA_DIR_PATH, 'meta'))
MODEL_DIR_PATH = path.abspath(path.join(PROJECT_ROOT, 'model'))

def check():
    print("FILE_PATH: ", FILE_PATH)
    print("UTIL_PATH: ", UTIL_PATH)
    print("SRC_DIR_PATH: ", SRC_DIR_PATH)
    print("PROJECT_ROOT: ", PROJECT_ROOT)
    print("DATA_DIR_PATH: ", DATA_DIR_PATH)
    print("ORIGIN_DATA_DIR_PATH: ", ORIGIN_DATA_DIR_PATH)
    print("ORIGIN_CONTENTS_DATA_DIR_PATH: ", ORIGIN_CONTENTS_DATA_DIR_PATH)
    print("ORIGIN_META_DATA_DIR_PATH: ", ORIGIN_META_DATA_DIR_PATH)
    print("PREPROCESSED_DATA_DIR_PATH: ", PREPROCESSED_DATA_DIR_PATH)
    print("PREPROCESSED_CONTENTS_DATA_DIR_PATH: ", PREPROCESSED_CONTENTS_DATA_DIR_PATH)
    print("PREPROCESSED_META_DATA_DIR_PATH: ", PREPROCESSED_META_DATA_DIR_PATH)
    print("MODEL_DIR_PATH: ", MODEL_DIR_PATH)


