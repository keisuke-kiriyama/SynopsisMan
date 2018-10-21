import os
import joblib
import json

from util.paths import OPT_SENTENCES_DATA_DIR_PATH, ACTIVE_NCODES_DIR_PATH
from util.corpus_accessor import CorpusAccessor
from data_supplier.opt_sentences_data_supplier import load

accessor = CorpusAccessor()

def big_genre(ncode):
    meta_file_path = accessor.create_meta_file_path(ncode)
    with open(meta_file_path, 'r') as f:
        data = json.load(f)
    return data['biggenre']

def save_ncodes(genre_name, ncodes):
    all_genre = ['general', 'love_story', 'fantasy', 'literature', 'sf', 'non_genre', 'other']
    if genre_name not in all_genre:
        raise ValueError("[ERROR] Genre does not exist")

    dir_path = os.path.join(ACTIVE_NCODES_DIR_PATH, genre_name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    file_path = os.path.join(dir_path, 'active_ncodes.txt')
    with open(file_path, 'wb') as train_f:
        joblib.dump(ncodes, train_f, compress=1)

def construct(threshold):
    if not (os.path.isdir(OPT_SENTENCES_DATA_DIR_PATH) and len(os.listdir(OPT_SENTENCES_DATA_DIR_PATH)) > 0):
        print("opt sentences data haven't constructed yet.")
        return

    file_names = os.listdir(OPT_SENTENCES_DATA_DIR_PATH)
    ncodes = [accessor.ncode_from_file_name(file_name) for file_name in file_names]

    # 付与された理想的なROUGEスコアで足切りをする
    general_ncodes = []
    love_story_ncodes = []
    fantasy_ncodes = []
    literature_ncodes = []
    sf_ncodes = []
    non_genre_ncodes = []
    other_ncodes = []
    for ncode in ncodes:
        data = load(ncode)
        if data['rouge']['r'] > threshold:
            general_ncodes.append(ncode)

            bg = big_genre(ncode)
            if bg == 1:
                love_story_ncodes.append(ncode)
            elif bg == 2:
                fantasy_ncodes.append(ncode)
            elif bg == 3:
                literature_ncodes.append(ncode)
            elif bg == 4:
                sf_ncodes.append(ncode)
            elif bg == 98:
                non_genre_ncodes.append(ncode)
            elif bg == 99:
                other_ncodes.append(ncode)

    save_ncodes('general', general_ncodes)
    save_ncodes('love_story', love_story_ncodes)
    save_ncodes('fantasy', fantasy_ncodes)
    save_ncodes('literature', literature_ncodes)
    save_ncodes('sf', sf_ncodes)
    save_ncodes('non_genre', non_genre_ncodes)
    save_ncodes('other', other_ncodes)




