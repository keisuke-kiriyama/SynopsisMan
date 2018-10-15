import joblib

from util.paths import ACTIVE_NCODES_FILE_PATH

def check_num():
    with open(ACTIVE_NCODES_FILE_PATH, 'rb') as f:
        data = joblib.load(f)
        print(len(data))

if __name__ == '__main__':
    check_num()
