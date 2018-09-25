# SynopsisMan
- A summarizer that automatically generates synopsis of novels.

## setup
```
pip install -r requirements.txt
```

## check setting path
- execute when you want to confirm the setting path.
```
python main.py check_path
```

## data preprocess
- Convert raw text into sentence-divided data (considering serifs too).
- Store scraped data in data/origin and execute this function.
- Then preprocessed data will store in data/preprocessed.
```
python main.py preprocess
```
- The data has the following structure. (example)
    - The list of contents is divided for each episode.
```json
{
  "n_code": "n0000aa",
  "sub_titles": [
    "sub title 1",
    "sub title 2",
  ],
  "contents": [
    ["sentence1", "sentence2"],
    ["sentence3", "sentence4", "sentence5"],
  ]
}
```

## training word embedding model
- You can train the word embedding model with the following command.
```
python main.py train_word_embedding_model
```

- If you need to refresh pathLineSentences files, execute following command.
```
python main.py train_word_embedding_model --refresh True
```

