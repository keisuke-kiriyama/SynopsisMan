from data_supplier.common_data_supplier import load_from_ncode
from util.paths import OPT_SENTENCES_DATA_DIR_PATH

def load(ncode):
    return load_from_ncode(ncode, dir_path=OPT_SENTENCES_DATA_DIR_PATH)
