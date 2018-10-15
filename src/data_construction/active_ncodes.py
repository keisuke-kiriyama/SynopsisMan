import os
import joblib

from util.paths import OPT_SENTENCES_DATA_DIR_PATH, ACTIVE_NCODES_FILE_PATH
from util.corpus_accessor import CorpusAccessor
from data_supplier.opt_sentences_data_supplier import load

accessor = CorpusAccessor()

def construct(threshold):
    if not (os.path.isdir(OPT_SENTENCES_DATA_DIR_PATH) and len(os.listdir(OPT_SENTENCES_DATA_DIR_PATH)) > 0):
        print("opt sentences data haven't constructed yet.")
        return

    file_names = os.listdir(OPT_SENTENCES_DATA_DIR_PATH)
    ncodes = [accessor.ncode_from_file_name(file_name) for file_name in file_names]

    # 付与された理想的なROUGEスコアで足切りをする
    active_ncodes = []
    for ncode in ncodes:
        data = load(ncode)
        if data['rouge']['r'] > threshold:
            active_ncodes.append(ncode)

    print('[INFO] saving active ncodes data...')
    with open(ACTIVE_NCODES_FILE_PATH, 'wb') as f:
        joblib.dump(active_ncodes, f, compress=1)




