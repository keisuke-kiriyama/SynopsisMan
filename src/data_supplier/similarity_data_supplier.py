import os
import joblib

from util.paths import SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH


def file_path_from_ncode(ncode):
    return os.path.join(SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH, ncode + '.txt')

def load_from_ncode(ncode):
    file_path = file_path_from_ncode(ncode)
    with open(file_path , 'rb') as f:
        data = joblib.load(f)
    return data







