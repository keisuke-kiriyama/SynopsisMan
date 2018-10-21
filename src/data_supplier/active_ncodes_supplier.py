import os
import joblib

from util.paths import ACTIVE_NCODES_DIR_PATH
from util.corpus_accessor import CorpusAccessor

corpus_accessor = CorpusAccessor()

def create_active_ncodes_file_path(genre):
    all_genre = ['general', 'love_story', 'fantasy', 'literature', 'sf', 'non_genre', 'other']
    if genre not in all_genre:
        raise ValueError("[ERROR] Genre does not exist")
    dir_path = os.path.join(ACTIVE_NCODES_DIR_PATH, genre)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    active_ncodes_file_path = os.path.join(dir_path, 'active_ncodes.txt')
    train_file_path = os.path.join(dir_path, 'train_ncodes.txt')
    test_file_path = os.path.join(dir_path, 'test_ncodes.txt')
    validation_file_path = os.path.join(dir_path, 'validation_ncodes.txt')
    return active_ncodes_file_path, train_file_path, test_file_path, validation_file_path

def get_active_ncodes(genre):
    """
    opt_sentenceのスコアにより足切りされたncodeを返す
    """
    all_genre = ['general', 'love_story', 'fantasy', 'literature', 'sf', 'non_genre', 'other']
    if genre not in all_genre:
        raise ValueError("[ERROR] Genre does not exist")

    file_path, _, _, _ = create_active_ncodes_file_path(genre)

    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            data = joblib.load(f)
            return data
    else:
        raise ValueError("[ERROR] active ncodes file does not exist")

def ncodes_train_test_split(genre, validation_size, test_size):
    """
    訓練データとテストデータのncodeを返す
    """
    _, train_ncodes_file_path, test_ncodes_file_path, validation_ncodes_file_path = create_active_ncodes_file_path(genre)

    if os.path.isfile(train_ncodes_file_path) \
            and os.path.isfile(test_ncodes_file_path) \
            and os.path.isfile(validation_ncodes_file_path):
        print('[INFO] loading splited ncodes data...')
        with open(train_ncodes_file_path, 'rb') as train_f:
            train_ncodes = joblib.load(train_f)
        with open(test_ncodes_file_path, 'rb') as test_f:
            test_ncodes = joblib.load(test_f)
        with open(validation_ncodes_file_path, 'rb') as validation_f:
            validation_ncodes = joblib.load(validation_f)

    else:
        active_ncodes = get_active_ncodes(genre)
        temp_ncodes = active_ncodes[:int(len(active_ncodes) * (1 - test_size))]
        test_ncodes = active_ncodes[int(len(active_ncodes) * (1 - test_size)):]
        train_ncodes = temp_ncodes[:int(len(temp_ncodes) * (1 - validation_size))]
        validation_ncodes = temp_ncodes[int(len(temp_ncodes) * (1 - validation_size)):]
        print('[INFO] saving splited ncodes data...')
        with open(train_ncodes_file_path, 'wb') as train_f:
            joblib.dump(train_ncodes, train_f, compress=3)
        with open(test_ncodes_file_path, 'wb') as test_f:
            joblib.dump(test_ncodes, test_f, compress=3)
        with open(validation_ncodes_file_path, 'wb') as validation_f:
            joblib.dump(validation_ncodes, validation_f, compress=3)

    print('[INFO] train ncodes count: {}'.format(len(train_ncodes)))
    print('[INFO] test ncodes count: {}'.format(len(test_ncodes)))
    print('[INFO] validation ncodes count: {}'.format(len(validation_ncodes)))
    return train_ncodes, test_ncodes, validation_ncodes
