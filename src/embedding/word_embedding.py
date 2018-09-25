import joblib
import os
import MeCab

from util.data_accessor import DataAccessor
from util.paths import PATH_LINE_SENTENCES_DIR_PATH

data_accessor = DataAccessor()

output_dir_path = PATH_LINE_SENTENCES_DIR_PATH

def create_path_line_sentences_files():
    """
    WordEmbeddingモデルを学習するためのファイルを作成する
    one sentence = one line, with words already preprocessed and separated by whitespace.
    作成されたファイルはdata/path_line_sentences下に保存される
    """
    tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd -Owakati')
    for i, ncode in enumerate(data_accessor.ncodes):
        if i % 50 == 0:
            print('progress: {:.1f}%, processing: {}'.format(i / len(data_accessor.ncodes) * 100, ncode))
        contents_liens = data_accessor.get_contents_lines(ncode)
        synopsis_lines = data_accessor.get_synopsis_lines(ncode)
        lines = contents_liens + synopsis_lines
        line_sentences = ''.join([tagger.parse(line) for line in lines])
        output_file_path = os.path.join(output_dir_path, ncode + '.txt')
        with open(output_file_path, 'w') as f:
            f.write(line_sentences)



