from data_supplier.common_data_supplier import load_from_ncode
from util.paths import WORD_EMBEDDING_AVG_VECTOR_CONTENTS_PATH

def load(ncode):
    return load_from_ncode(ncode, dir_path=WORD_EMBEDDING_AVG_VECTOR_CONTENTS_PATH)
