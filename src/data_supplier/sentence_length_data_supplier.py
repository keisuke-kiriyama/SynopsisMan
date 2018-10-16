from data_supplier.common_data_supplier import load_from_ncode
from util.paths import SENTENCE_LENGTH_CONTENTS_PATH

def load(ncode):
    return load_from_ncode(ncode, dir_path=SENTENCE_LENGTH_CONTENTS_PATH)
