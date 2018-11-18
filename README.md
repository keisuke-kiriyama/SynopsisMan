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
- You need to set data directory path
- following command is example
```
export DATA_DIR_PATH=~/src/SynopsisMan/data
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
- When embedding model has constructed, execute following command
```
python main.py construct_embedding_matrix
``` 

## Training data construction

### Similarity between contents and synopsis

- Execute following command, then data on the similarity between the contents and the synopsis is constructed
- based on cos similarity
```
python main.py construct_similarity_data
```
- based on rouge score
```
python main.py construct_rouge_similarity_data
```

### Average of word embedding vectors	
- Execute following command, then average of word embedding vectors data is constructed.	
```	
python main.py construct_word_embedding_avg_vector	
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
- Execute following command, then sentence length data is constructed.
```
python main.py construct_sentence_length_data
```

### Opt Sentences data
- It is data such as the ROUGE score of the summary created by selecting the ideal sentence with the training data
- This data has the following structure.
```
{
    opt_sentence_index: np.array,
    threshold: float,
    rouge:
        {
        f: float,
        r: float,
        p: float
        }
}
``` 
- Execute following command, then this data is constructed.
    - The parameter represents the following information
        - --short_rate: summary rate of short stories.
        - --long_rate: summary rate of long stories.
        - --min_sentence_count: Maximum number of sentences in synopsis
        - --max_sentence_count: Minimum number of sentences in synopsis
```
python main.py construct_opt_sentences_data --short_rate 0.051 --long_rate 0.013 --min_sentence_count 1 --max_sentence_count 6
```

- After construct opt sentences data, by doing following command, ncodes to be used is determined.
    - --threshold: Lower limit of the score given to the novel.
```
python main.py construct_active_ncodes_data --threshold 0.3
```

## Train neural net model
- Execute following command, then neural net model is trained.
#### DNN
```
python main.py dnn_summarizer_fit -g general -i cos_sim --position --serif --person --sentence_length
```
#### LSTM
```
python main.py lstm_summarizer_fit -g general -i cos_sim --position --serif --person --sentence_length
```

#### option
- --genre or -g: genre of novel
    - general
    - love_story
    - fantasy
    - literature
    - sf
    - non_genre
    - other
- --importance or -i : importance score
    - cos_sim
    - rouge
    
- feature
    - --embedding_vector: average vector of word embedding vectors 
    - --position: position of sentence
    - --serif: is include serif
    - --person: is include person name 
    - --sentence_length: length of sentence
    
## Evaluation
### ROUGE

- If you want to evaluate synopsis generation by rouge scocre, execute following command.
```
python main.py evaluate_rouge_score -g general -i cos_sim --position --serif --person --sentence_length
```

- Option is equal to fit option.
