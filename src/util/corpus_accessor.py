import os
import json
from itertools import chain

from util import paths
from util.text_processor import get_wakati_lines

class CorpusAccessor:

    def __init__(self):
        # PATHS
        self.contents_data_dir_path = paths.PREPROCESSED_CONTENTS_DATA_DIR_PATH
        self.meta_data_dir_path = paths.PREPROCESSED_META_DATA_DIR_PATH
        if not os.path.isdir(self.contents_data_dir_path) or not os.path.isdir(self.meta_data_dir_path): return
        self.contents_file_paths = [os.path.join(self.contents_data_dir_path, file_name) for file_name in os.listdir(self.contents_data_dir_path) if not file_name == '.DS_Store']
        self.meta_file_paths = [os.path.join(self.meta_data_dir_path, file_name) for file_name in os.listdir(self.meta_data_dir_path) if not file_name == '.DS_Store']
        self.ncodes = [self.ncode_from_file_path(file_path) for file_path in self.contents_file_paths]

        self.active_ncodes = self.get_active_ncodes()

    def ncode_from_file_path(self, file_path):
        """
        ファイルpathからncodeを返却する
        """
        ncode = file_path.split('/')[-1].split('.')[0]
        if '_meta' in ncode:
            ncode = ncode.replace('_meta', '')
        return ncode

    def ncode_from_file_name(self, file_name):
        """
        ファイル名からncodeを返却する
        """
        ncode = file_name.split('.')[0]
        if '_meta' in ncode:
            ncode = ncode.replace('_meta', '')
        return ncode

    def create_contents_file_path(self, ncode):
        """
        ncodeから本文の情報が格納されているファイルパスを作成する
        """
        dir_path = self.contents_data_dir_path
        return os.path.join(dir_path, ncode + '.json')

    def create_meta_file_path(self, ncode):
        """
        ncodeからメタ情報が格納されているファイルパスを作成する
        """
        dir_path = self.meta_data_dir_path
        return os.path.join(dir_path, ncode+'_meta.json')

    def load(self, file_path):
        json_file = open(file_path, 'r')
        data = json.load(json_file)
        json_file.close()
        return data

    def get_contents_lines(self, ncode):
        """
        本文全文のリストを返却
        :param contents_file_path: str
        :return: list
        """
        contents_file_path = self.create_contents_file_path(ncode=ncode)
        if not contents_file_path in self.contents_file_paths:
            print(contents_file_path)
            print('nothing ncode')
            return
        return list(chain.from_iterable(self.load(contents_file_path)['contents']))

    def get_synopsis_lines(self, ncode):
        """
        あらすじの文のリストを返却
        :param synopsis_file_path: str
        :return: list
        """
        meta_file_path = self.create_meta_file_path(ncode=ncode)
        if not meta_file_path in self.meta_file_paths:
            print('nothing ncode')
            return
        return self.load(meta_file_path)['story']

    def get_wakati_contents_lines(self, ncode):
        """
        本文の各文を分かち書きしたリストを取得
        :param ncode: str
        :return: list
        """
        contents_lines = self.get_contents_lines(ncode=ncode)
        wakati_contents_lines = get_wakati_lines(contents_lines)
        return wakati_contents_lines

    def get_wakati_synopsis_lines(self, ncode):
        """
        あらすじの各分を分かち書きしたリストを取得
        """
        synopsis_lines = self.get_synopsis_lines(ncode=ncode)
        wakati_synopsis_lines = get_wakati_lines(synopsis_lines)
        return wakati_synopsis_lines

    def is_long(self, ncode):
        """
        長編小説: True, 短編小説: False
        """
        file_path = self.create_meta_file_path(ncode)
        data = self.load(file_path)
        return data['noveltype'] == 1

    def is_end(self, ncode):
        """
        完結済み: True, 連載中: False
        """
        file_path = self.create_meta_file_path(ncode)
        data = self.load(file_path)
        return data['end'] == 0

    def get_active_ncodes(self):
        """
        データの構築状況によりアクティブなncodeを返す
        """
        # 理想的な文選択のデータが構築済みの場合
        if len(os.listdir(os.path.join(paths.OPT_SENTENCES_DIR_PATH, 'short_5.1_long_1.3_min_1_max_6'))):
            return self.__active_ncodes_from_file_path(os.path.join(paths.OPT_SENTENCES_DIR_PATH, 'short_5.1_long_1.3_min_1_max_6'))

        # 類似度のデータが構築済みの際
        if len(os.listdir(paths.SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH)) > 0:
            return self.__active_ncodes_from_file_path(paths.SIMILARITY_BETWEEEN_CONTENTS_AND_SYNOPSIS_SENTENCE_DIR_PATH)

        # スクレイピングしたデータの前処理が完了している際
        if len(os.listdir(paths.PREPROCESSED_CONTENTS_DATA_DIR_PATH)) > 0:
            return self.__active_ncodes_from_file_path(paths.PREPROCESSED_CONTENTS_DATA_DIR_PATH)

        if len(os.listdir(paths.ORIGIN_CONTENTS_DATA_DIR_PATH)) > 0:
            return self.__active_ncodes_from_file_path(paths.ORIGIN_CONTENTS_DATA_DIR_PATH)

        print("Data haven't stored yet")
        return None

    def __active_ncodes_from_file_path(self, path):
        return [self.ncode_from_file_name(file_name) for file_name in os.listdir(path) if not file_name == '.DS_Store']




