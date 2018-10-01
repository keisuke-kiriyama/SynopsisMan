import os
import MeCab
import logging
from gensim.models import word2vec
from gensim.models.word2vec import PathLineSentences

from util.corpus_accessor import CorpusAccessor
from util.paths import PATH_LINE_SENTENCES_DIR_PATH, WORD_EMBEDDING_MODEL_PATH
from util.text_processor import wakati_tagger

data_accessor = CorpusAccessor()

output_dir_path = PATH_LINE_SENTENCES_DIR_PATH
embedding_model_path = WORD_EMBEDDING_MODEL_PATH

# PROPERTY
embedding_size = 200
embedding_window = 15
embedding_min_count = 0
embedding_sg = 0

def create_path_line_sentences_files():
    """
    WordEmbeddingモデルを学習するためのファイルを作成する
    one sentence = one line, with words already preprocessed and separated by whitespace.
    作成されたファイルはdata/path_line_sentences下に保存される
    """
    tagger = wakati_tagger()
    for i, ncode in enumerate(data_accessor.ncodes):
        if i % 30 == 0:
            print('progress: {:.1f}%'.format(i / len(data_accessor.ncodes) * 100))
        contents_liens = data_accessor.get_contents_lines(ncode)
        synopsis_lines = data_accessor.get_synopsis_lines(ncode)
        lines = contents_liens + synopsis_lines
        line_sentences = ''.join([tagger.parse(line) for line in lines])
        output_file_path = os.path.join(output_dir_path, ncode + '.txt')
        with open(output_file_path, 'w') as f:
            f.write(line_sentences)


def train_word_embedding_model():
    """
    WordEmbeddingModelを学習する
    """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = PathLineSentences(PATH_LINE_SENTENCES_DIR_PATH)
    model = word2vec.Word2Vec(sentences,
                              size=embedding_size,
                              window=embedding_window,
                              min_count=embedding_min_count,
                              sg=embedding_sg)
    model.save(embedding_model_path)

def test_word_embedding_model(test_word):
    """
    WordEmbeddingModelのテスト
    """
    model = word2vec.Word2Vec.load(embedding_model_path)
    results = model.wv.most_similar(positive=[test_word])
    for result in results:
        print(result)