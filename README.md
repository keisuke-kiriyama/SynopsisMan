# SynopsisMan
- A summarizer that automatically generates synopsis of novels.

## Setup
```
pip install -r requirements.txt
```

## Check setting path
- execute when you want to confirm the setting path.
```
python main.py check_path
```

## Data preprocess
- Convert raw text into sentence-divided data (considering serifs too).
- Store scraped data in `data/origin` and execute this function.
- Then preprocessed data will store in `data/preprocessed`.
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

## Training word embedding model
- You can train the word embedding model with the following command.
```
python main.py word_embedding
```

- If you need to refresh pathLineSentences files, execute following command.
```
python main.py word_embedding --refresh
```
- then trained word embedding model will be saved in `/model/word_embedding/word_embedding.model`

- If you want to test word embedding model, then execute following command.
```
python main.py test_word_embedding -w 部屋
```

## Training data construction
### Average of word embedding vectors
- execute following command, then average of word embedding vectors data is constructed.
```
python main.py construct_word_embedding_avg_vector
```