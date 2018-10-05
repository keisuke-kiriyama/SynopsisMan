import os
from util.paths import SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH
from util.corpus_accessor import CorpusAccessor


class SimilarityDataSupplier():

    def __init__(self):
        self.data_dir_path = SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH
        self.file_paths = [os.path.join(SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH, file_name)
                           for file_name in os.listdir(SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH)
                           if not file_name == '.DS_Store']

    def file_path_from_ncode(self, ncode):
        return os.path.join(self.data_dir_path, ncode + '.txt')

if __name__ == '__main__':
    s = SimilarityDataSupplier()
    ncode = 'abdc'
    s.file_path_from_ncode(ncode)





