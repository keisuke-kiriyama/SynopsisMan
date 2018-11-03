import sys
from rouge import Rouge
from util.corpus_accessor import CorpusAccessor
from util.text_processor import wakati
from summarizer.lstm_summarizer import LSTMSummarizer
from summarizer.dnn_summarizer import DNNSummarizer
from data_supplier.lstm_vector_supplier import LSTMVectorSupplier
from data_supplier.dnn_vector_supplier import DNNVectorSupplier
from evaluate import lead_synopsis
from evaluate import opt_synopsis
from evaluate import random_synopsis

corpus_accessor = CorpusAccessor()

def evaluate(genre='general',
             importance='cos_sim',
             use_data_of_position_of_sentence=False,
             use_data_of_is_serif=False,
             use_data_of_is_include_person=False,
             use_data_of_sentence_length=False):
    dnn_summarizer = DNNSummarizer()
    dnn_vector_supplier = DNNVectorSupplier(genre,
                                 importance,
                                 use_data_of_position_of_sentence=use_data_of_position_of_sentence,
                                 use_data_of_is_serif=use_data_of_is_serif,
                                 use_data_of_is_include_person=use_data_of_is_include_person,
                                 use_data_of_sentence_length=use_data_of_sentence_length)
    dnn_summarizer.set_supplier(dnn_vector_supplier)

    lstm_summarizer = LSTMSummarizer()
    lstm_vector_supplier = LSTMVectorSupplier(genre,
                                  importance,
                                  use_data_of_position_of_sentence=use_data_of_position_of_sentence,
                                  use_data_of_is_serif=use_data_of_is_serif,
                                  use_data_of_is_include_person=use_data_of_is_include_person,
                                  use_data_of_sentence_length=use_data_of_sentence_length)
    lstm_summarizer.set_supplier(lstm_vector_supplier)

    test_ncodes = lstm_vector_supplier.test_ncodes
    total = len(test_ncodes)
    print('[INFO] test ncodes count: ', total)

    opt_scores = []               # 類似度上位から文選択(理論上の上限値)
    lead_scores = []              # 文章の先頭からoptの文数分選択
    randoms_scores = []            # ランダムに文を選択
    dnn_scores = []                # DNNによるあらすじ
    lstm_scores = []               # LSTMによるあらすじ

    sys.setrecursionlimit(20000)
    rouge = Rouge()
    for i, ncode in enumerate(test_ncodes):
        ncode = 'n0013da'
        print('[INFO] processing ncode: ', ncode)
        print('[INFO] progress: {:.1f}%'.format(i / total * 100))

        ref = wakati(''.join(corpus_accessor.get_synopsis_lines(ncode)))
        opt = wakati(opt_synopsis.generate(ncode))
        lead = wakati(lead_synopsis.generate(ncode))
        random = wakati(random_synopsis.generate(ncode))
        dnn_hyp = wakati(dnn_summarizer.generate(ncode))
        lstm_hyp = wakati(lstm_summarizer.generate(ncode))

        print(ref)
        print(opt)

        return














if __name__ == '__main__':
    evaluate()