from data_supplier.common_data_supplier import load_from_ncode
from util.paths import POSITION_OF_SENTENCE_CONTENTS_DIR_PATH

def load(ncode):
    return load_from_ncode(ncode, dir_path=POSITION_OF_SENTENCE_CONTENTS_DIR_PATH)
