from data_supplier.common_data_supplier import load_from_ncode
from util.paths import IS_INCLUDE_PERSON_CONTENTS_PATH

def load(ncode):
    return load_from_ncode(ncode, dir_path=IS_INCLUDE_PERSON_CONTENTS_PATH)
