# SynopsisMan
- A summarizer that automatically generates synopsis of novels

## setup
```
pip install -r requirements.txt
```

## check setting path
- execute when you want to confirm the setting path
```
python main.py check_path
```

## data preprocess

## commit rule
Please attach one of them to the beginning of the commit comment.

| Emoji | code | timing | example |
| :---: | :---: | :--- | --- |
| :heavy_plus_sign: | `:heavy_plus_sign:` | 機能を追加したとき | フィルタ機能を追加 |
| :wrench: | `:wrench:` | 機能追加とまでは言えない仕様変更をしたとき | ダイアログの表示条件を変更 |
| :art: | `:art:` | コードの可読性や保守性を改善したとき | クラス分割、Group分け |
| :racehorse: | `:racehorse:` | パフォーマンスを改善した時 | スレッド分割、ビルド時間短縮 |
| :bug: | `:bug:` | バグを修正した時 | クラッシュバグ修正 |
| :arrow_up: | `:arrow_up:` | アプリやライブラリのバージョンアップをしたとき | アプリのバージョン番号変更、pod update |
| :arrow_down: | `:arrow_down:` | アプリやライブラリのバージョンダウンをしたとき | Parse SDKの最新版に不具合があってバージョンダウン |
| :bird: | `:bird:` | Swift化 | Objective-cののSwift化 |
| :fire: | `:fire:` | コードやファイルを削除したとき | 未使用のクラス削除 |
| :package:  | `:package:` | ファイルを移動したとき | ディレクトリ構成変更 |
| :shirt: | `:shirt:` | warningを取り除いたとき | SwiftLintの警告解消 |
| :white_check_mark: | `:white_check_mark:` | テストを追加・編集したとき | Serviceのテストメソッド追加 |
| :memo: | `:memo:` | ドキュメントを書いたとき | CONTRIBUTING.md編集 | :green_heart: | `:green_heart:` | Jenkins用に何かを変更したとき | scriptsディレクトリ内のファイルを差し替え |
| :ok: | `:ok:` | なにかokな変更をしたとき | ラベル変更など、どれにも当てはまらないがとにかく良くなる変更 |

## 例
```
:bird: Convert SyncService to swift
```