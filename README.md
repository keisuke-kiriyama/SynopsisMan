# SynopsisMan
- A summarizer that automatically generates synopsis of novels.

## Setup
```
pip install -r requirements.txt
```
- You need to set mecab-ipadic-NEologd dictionary path
- following command is example
```
export NEOLOGD_PATH=/usr/local/lib/mecab/dic/mecab-ipadic-neologd
```


## Check setting path
- Execute when you want to confirm the setting path.
```
python main.py check_path
```

## Data preprocess
- First, you need to execute following command.
```
python main.py data_mkdir
```
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
- Then trained word embedding model will be saved in `/model/word_embedding/word_embedding.model`

- If you want to test word embedding model, then execute following command.
```
python main.py test_word_embedding -w 部屋
```

## Training data construction
### Average of word embedding vectors
- Execute following command, then average of word embedding vectors data is constructed.
```
python main.py construct_word_embedding_avg_vector
```

### Similarity between contents and synopsis
- Execute following command, then data on the similarity between the contents and the synopsis is constructed
```
python main.py construct_similarity_data:
```

### Position of sentence
- Execute following command, then data of position of sentence is constructed.
```
python main.py construct_position_of_sentence_data
```

### Is Serif?
- Execute following command, then data which Data indicating whether each sentence is serif is constructed.
```
python main.py construct_is_serif_data
```

### Is include person name?
- Execute following command, then data representing whether or not a person's name is included is constructed
```
python main.py construct_is_include_person_data
```

### Sentence length
- Execute following command, then entence length data is constructed.
```
python main.py construct_sentence_length_data
```
