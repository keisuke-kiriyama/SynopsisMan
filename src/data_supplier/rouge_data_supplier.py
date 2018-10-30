from data_supplier.common_data_supplier import load_from_ncode
from util.paths import ROUGE_BETWEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH

def load(ncode):
    return load_from_ncode(ncode, dir_path=ROUGE_BETWEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH)
