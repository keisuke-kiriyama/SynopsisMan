import os
import json

from util.corpus_accessor import CorpusAccessor
from util import paths
from util.text_processor import cleaning, convert_serif_marker
from preprocess.contents import splited_sentences

meta_origin_file_paths = [os.path.join(paths.ORIGIN_META_DATA_DIR_PATH, file_name)
                          for file_name in os.listdir(paths.ORIGIN_META_DATA_DIR_PATH)
                          if not file_name == '.DS_Store']

# Data Accessor
accessor = CorpusAccessor()

def remove_publicity_sentences(sentences):
    """
    宣伝用の文を除去する
    """
    pablicity_words = [
        'ランキング',
        '日間',
        '月間',
        '累計',
        'アルファポリス',
        '完結',
        '完結済',
        '連載'
        '連載中',
        'ハイファンタジー',
        'PV',
        '休載',
        'http',
        '本編',
        '番外編',
        'フィクション',
        '携帯版',
        '投稿予定',
        'サブタイ',
        '改稿',
        '改稿済',
        '感想',
        'シリーズ',
        '更新',
        '掲載'
        '不定期'
        '改編',
        '短編',
        '短篇',
        '中編',
        '長編',
        '欠番編',
        'オーディオブック',
        '作者',
        'コメディー',
        'お楽しみください',
        '前半',
        '後半',
        '読めます',
        '公開',
        'FA',
        '挿絵',
        '設定',
        'あらすじ',
        '書きなおす',
        '発売',
        '転載',
        '[PG12]',
        'モーニングスター大賞',
        '応援よろしくお願いいたします',
        'タグ',
        '書籍化',
        'スピンオフ',
        '重複展開',
        'ご了承',
        '読んでくださる',
        'タイトルを変更',
        '途中で放棄',
        '掲載中',
        '本文の書き換え',
        'ブログ',
        '他サイト',
        'ありがとうございました',
        '話予定',
        'そちらを読んでから',
        '前作',
        'オススメ',
        '残酷描写',
        '差別表現',
        'ご愛読',
        'R15',
        'Ｒ１５',
        '実在のものとは関係ありません',
        'ネタバレ',
        'カクヨム',
        '処女作',
        '執筆中',
        'pixiv',
        '２ｃｈ',
        '2ch',
        '鬱展開',
        'チート',
        '読みにくい'
    ]
    removed_sentences = []
    for sentence in sentences:
        is_publicity_sentence = True in [pablicity_word in sentence for pablicity_word in pablicity_words]
        if not is_publicity_sentence and len(sentence) > 1:
            removed_sentences.append(sentence)
        else:
            print(sentence)
    return removed_sentences

def preprocess_synopsis(synopsis):
    """
    あらすじの前処理
    :param synopsis: str
    :return: list
    """
    if not synopsis[-1] in ['。', '？', '！']:
        synopsis += '。'
    synopsis = cleaning(synopsis)
    synopsis = convert_serif_marker(synopsis)
    synopsis_sentences = splited_sentences(synopsis)
    preprocessed_sentences = remove_publicity_sentences(synopsis_sentences)
    return preprocessed_sentences

def execute():
    """
    あらすじの文分割を行い、宣伝用の文などを除去する
    スクレイピングしたデータをmeta_originにまとめ、この関数を回す
    前処理されたファイルはmetaディレクトリ下に保存される
    """
    for i, meta_origin_file_path in enumerate(meta_origin_file_paths):
        ncode = accessor.ncode_from_file_path(meta_origin_file_path)
        if i % 50 == 0:
            print('progress: {:.1f}%, processing: {}'.format(i / len(meta_origin_file_paths) * 100, ncode))
        meta_data = accessor.load(file_path=meta_origin_file_path)
        meta_data['story'] = preprocess_synopsis(meta_data['story'])
        output_file_path = accessor.create_meta_file_path(ncode=ncode)
        with open(output_file_path, 'w') as f:
            json.dump(meta_data, f, ensure_ascii=False)

